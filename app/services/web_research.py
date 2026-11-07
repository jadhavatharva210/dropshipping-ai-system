"""Live web research: fetch product pages and extract structured facts.

Agent 1's other adapters are offline on purpose. This one is different - it
talks to real supplier and competitor websites - so it is built to a stricter
standard than the rest of the codebase:

**It never invents a number.** A price is reported only when the page itself
declares one in machine-readable structured data (JSON-LD ``Product``/``Offer``
or an OpenGraph product price tag). A page with no such tag yields
``price_usd = None`` and ``escalation_reason`` explaining why, never a guess
scraped out of prose. Structured prices are ``VERIFIED`` because they are the
merchant's own machine-readable claim, quoted from the URL in
``source_reference``; anything weaker (a title, a description, a price found
only in text) is ``ESTIMATED`` at best.

**It is SSRF-safe by construction.** ``file://``, ``javascript:``, ``data:``,
loopback, link-local (169.254.x, cloud metadata), private and reserved address
ranges are rejected *before* a request is built. Validation happens on the URL
and again on the resolved DNS addresses, so a public hostname pointing at
``127.0.0.1`` is still refused.

**It is light.** Only the stdlib (``html.parser``, ``json``, ``re``) plus the
already-required ``httpx``. No headless browser, no BeautifulSoup, no lxml - so
it runs comfortably inside the project's 8 GB RAM budget.

Domain vocabulary mirrors what real stores actually emit:

* Shopify/WooCommerce JSON-LD: ``{"@type": "Product", "offers": {...}}``,
  sometimes wrapped in ``@graph``, with ``offers`` as a list, ``availability``
  as ``https://schema.org/InStock`` (or a bare ``InStock``), and ``price`` as a
  number, a string, or a ``priceSpecification``.
* OpenGraph: ``product:price:amount`` / ``og:price:amount``,
  ``product:price:currency`` / ``og:price:currency``, ``og:title``,
  ``og:availability``, ``meta[name="description"]``.
"""

from __future__ import annotations

import html as html_module
import ipaddress
import json
import re
import socket
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlsplit

import httpx

from app.core.config import settings
from app.schemas.product_research import DataProvenance

__all__ = [
    "ALLOWED_SCHEMES",
    "BLOCKED_HOSTNAMES",
    "BROWSER_HEADERS",
    "ExtractedWebProduct",
    "UnsourcedPriceError",
    "UnsafeUrlError",
    "WebResearchError",
    "WebResearchService",
    "WebSearchHit",
    "extract_product_from_html",
    "validate_public_url",
]

#: Only these schemes are ever fetched. Everything else (``file:``, ``ftp:``,
#: ``javascript:``, ``data:``) is rejected before a request exists.
ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

#: Hostnames that never point at a public merchant, plus the ones that point at
#: *this* machine or the cloud metadata service.
BLOCKED_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
        "instance-data",
        # Cloud metadata service IPs, reachable without any DNS lookup.
        "169.254.169.254",
        "fd00:ec2::254",
    }
)

#: Suffixes that only resolve inside a private network.
BLOCKED_HOST_SUFFIXES: tuple[str, ...] = (
    ".local",
    ".internal",
    ".intranet",
    ".lan",
    ".home",
    ".corp",
)

#: Realistic browser headers. Many stores return a stripped-down page (or 403)
#: to a bare ``python-httpx`` user agent, and the structured data is what we
#: are after, so the request looks like a normal browser page load.
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
}

#: Refuse absurdly large pages: we only need the ``<head>`` and the JSON-LD
#: blocks, and a 50 MB download is a denial-of-service on a small machine.
MAX_RESPONSE_BYTES: int = 2_000_000

#: Only the first slice of the body is parsed, for the same reason.
MAX_HTML_CHARS: int = 500_000

DEFAULT_SEARCH_RESULTS: int = 5
MAX_SEARCH_RESULTS: int = 10

#: Search backends. ``none`` never touches the network; ``duckduckgo`` parses
#: the public HTML endpoint (no key, no cost); ``custom`` calls a configured
#: JSON API. The default is ``none`` so nothing implicit leaves the machine.
SEARCH_PROVIDERS: tuple[str, ...] = ("none", "duckduckgo", "custom")

DUCKDUCKGO_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"

#: Hosts a *search* result may never point at, on top of the SSRF checks.
_BLOCKED_RESULT_SCHEMES = frozenset({"http", "https"})


class WebResearchError(RuntimeError):
    """Base class for live-research failures."""


class UnsafeUrlError(WebResearchError, ValueError):
    """The URL was rejected before any network call (SSRF guard)."""


class UnsourcedPriceError(WebResearchError):
    """A page carried no structured price, so none could be reported.

    Raised only by :meth:`WebResearchService.require_price`; the fetch itself
    returns an ``ExtractedWebProduct`` with ``price_usd = None`` so a caller can
    decide whether a missing price is fatal or merely a downgrade.
    """


@dataclass(frozen=True)
class WebSearchHit:
    """One search result: a candidate URL plus the snippet that suggested it."""

    url: str
    title: str
    snippet: str = ""
    source: str = "search"
    rank: int = 0


@dataclass(frozen=True)
class ExtractedWebProduct:
    """What a product page actually declared, and nothing more.

    ``price_usd`` is ``None`` unless the page carried a structured price.
    ``provenance`` is ``VERIFIED`` only for a structured price quoted from
    ``source_reference``; a page that yielded some facts but no structured price
    is ``ESTIMATED``.
    """

    url: str
    product_name: str = ""
    description: str = ""
    brand: str = ""
    seller: str = ""
    price_usd: float | None = None
    currency: str = ""
    #: ``True`` / ``False`` from structured data, ``None`` when unconfirmed.
    in_stock: bool | None = None
    availability_raw: str = ""
    provenance: DataProvenance = DataProvenance.SIMULATED
    source_reference: str = ""
    extraction_source: str = ""
    #: URLs discovered in the page's own structured data (offer URLs).
    offer_urls: tuple[str, ...] = ()
    #: Why this page is less than fully sourced (empty when it is fully sourced).
    escalation_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_structured_price(self) -> bool:
        return self.price_usd is not None and self.provenance is DataProvenance.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "product_name": self.product_name,
            "description": self.description,
            "brand": self.brand,
            "seller": self.seller,
            "price_usd": self.price_usd,
            "currency": self.currency,
            "in_stock": self.in_stock,
            "availability_raw": self.availability_raw,
            "provenance": self.provenance.value,
            "source_reference": self.source_reference,
            "extraction_source": self.extraction_source,
            "offer_urls": list(self.offer_urls),
            "escalation_reason": self.escalation_reason,
            "extra": dict(self.extra),
        }


# ══════════════════════════════════════════════════════════════════════════
# SSRF guard
# ══════════════════════════════════════════════════════════════════════════


def _ip_is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True only for globally routable unicast addresses.

    ``is_private`` alone is not enough: it misses link-local (169.254/16, the
    cloud metadata range), shared CGNAT space and reserved blocks, all of which
    ``is_global`` correctly excludes.
    """
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False
    return ip.is_global


def validate_public_url(url: str) -> str:
    """Return the URL if it is safe to fetch, else raise :class:`UnsafeUrlError`.

    Checks, in order: presence, scheme allow-list, hostname block-list (literal
    ``localhost`` and friends), literal-IP range check, then DNS resolution of
    the hostname. Each failure names the rule that rejected it so an operator
    can see the request was refused deliberately.
    """
    if not isinstance(url, str) or not url.strip():
        raise UnsafeUrlError("URL must be a non-empty string")

    text = url.strip()
    parts = urlsplit(text)
    scheme = (parts.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise UnsafeUrlError(
            f"Refusing to fetch scheme {scheme or '(none)'!r}: only http and https are allowed"
        )

    hostname = (parts.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise UnsafeUrlError(f"URL {text!r} has no hostname")
    if hostname in BLOCKED_HOSTNAMES:
        raise UnsafeUrlError(
            f"Refusing to fetch {hostname!r}: loopback, local and cloud-metadata "
            "hostnames are blocked"
        )
    if any(hostname.endswith(suffix) for suffix in BLOCKED_HOST_SUFFIXES):
        raise UnsafeUrlError(
            f"Refusing to fetch {hostname!r}: private-network hostnames are blocked"
        )

    # A literal address needs no DNS: judge it directly.
    try:
        literal = ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        literal = None
    if literal is not None:
        if not _ip_is_public(literal):
            raise UnsafeUrlError(
                f"Refusing to fetch {hostname}: private, loopback, link-local and "
                "reserved addresses are blocked"
            )
        return text

    # A hostname is only safe if *every* address it resolves to is public -
    # otherwise a public-looking domain could point at 127.0.0.1.
    try:
        infos = socket.getaddrinfo(hostname, parts.port or (443 if scheme == "https" else 80))
    except socket.gaierror as exc:
        raise UnsafeUrlError(
            f"Refusing to fetch {hostname!r}: hostname did not resolve ({exc})"
        ) from exc

    for info in infos:
        address = info[4][0]
        try:
            resolved = ipaddress.ip_address(str(address).split("%", 1)[0])
        except ValueError:  # pragma: no cover - getaddrinfo returns valid IPs
            raise UnsafeUrlError(f"Refusing to fetch {hostname!r}: unparseable address") from None
        if not _ip_is_public(resolved):
            raise UnsafeUrlError(
                f"Refusing to fetch {hostname!r}: it resolves to the non-public "
                f"address {resolved}"
            )
    return text


# ══════════════════════════════════════════════════════════════════════════
# HTML parsing
# ══════════════════════════════════════════════════════════════════════════

_CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "A$": "AUD",
    "C$": "CAD",
}


class _PageParser(HTMLParser):
    """Collects the ``<meta>`` tags and JSON-LD blocks from a page.

    Deliberately tiny: it records exactly the two structured sources this
    service trusts, plus the ``<title>`` for a human-readable fallback name.
    Nothing is inferred from visible prose.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.json_ld_blocks: list[str] = []
        self.title: str = ""
        self._in_script = False
        self._script_is_json_ld = False
        self._script_buffer: list[str] = []
        self._in_title = False
        self._title_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name.lower(): (value or "") for name, value in attrs}
        if tag == "meta":
            key = (
                attributes.get("property")
                or attributes.get("name")
                or attributes.get("itemprop")
                or ""
            ).strip().lower()
            content = attributes.get("content")
            if key and content is not None and key not in self.meta:
                self.meta[key] = content.strip()
        elif tag == "script":
            script_type = attributes.get("type", "").strip().lower()
            self._in_script = True
            self._script_is_json_ld = script_type == "application/ld+json"
            self._script_buffer = []
        elif tag == "title":
            self._in_title = True
            self._title_buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_script:
            if self._script_is_json_ld:
                block = "".join(self._script_buffer).strip()
                if block:
                    self.json_ld_blocks.append(block)
            self._in_script = False
            self._script_is_json_ld = False
            self._script_buffer = []
        elif tag == "title" and self._in_title:
            self.title = "".join(self._title_buffer).strip()
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self._script_buffer.append(data)
        elif self._in_title:
            self._title_buffer.append(data)


def _parse_money(raw: Any) -> float | None:
    """Parse a price out of whatever shape the page used.

    Handles ``29.99``, ``"29.99"``, ``"$29.99"``, ``"29,99"`` and
    ``"1,299.00"``. Returns ``None`` for anything that is not a positive number
    - a missing price must stay missing.
    """
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return round(value, 2) if value > 0 else None
    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if not text:
        return None
    # Drop currency symbols/codes, keep digits, separators and signs.
    cleaned = re.sub(r"[^\d.,-]", "", text)
    if not cleaned:
        return None
    if "," in cleaned and "." in cleaned:
        # 1,299.00 -> thousands separator before a decimal point
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        # 29,99 -> decimal comma (European storefronts)
        whole, _, frac = cleaned.partition(",")
        cleaned = f"{whole}.{frac}" if len(frac) in (1, 2) else f"{whole}{frac}"
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return round(value, 2) if value > 0 else None


def _parse_currency(raw: Any, *, default: str = "") -> str:
    if isinstance(raw, str) and raw.strip():
        value = raw.strip()
        if value.upper() == value and len(value) in (3, 4):
            return value.upper()
        return _CURRENCY_SYMBOLS.get(value, value.upper() if len(value) == 3 else value)
    return default


_AVAILABILITY_PATTERNS: tuple[tuple[re.Pattern[str], bool], ...] = (
    (re.compile(r"out\s?of\s?stock|outofstock|sold\s?out|discontinued|back\s?order"), False),
    (re.compile(r"in\s?stock|instock|available|limited\s?availability|pre\s?order"), True),
)


def _parse_availability(raw: Any) -> bool | None:
    """Map a schema.org/OpenGraph availability string to a tri-state flag.

    ``None`` means "the page did not say", which is not the same as out of
    stock - guessing either way would be a fabrication.
    """
    if isinstance(raw, bool):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().lower()
    # Check "out of stock" first: "https://schema.org/OutOfStock" also contains
    # no "in stock", but "InStock" is a substring of nothing dangerous - order
    # still matters for phrases like "available: out of stock".
    for pattern, value in _AVAILABILITY_PATTERNS:
        if pattern.search(text):
            return value
    return None


def _as_text(raw: Any) -> str:
    """Flatten a schema.org text-ish value (str, list, or ``{"name": ...}``)."""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return str(raw)
    if isinstance(raw, dict):
        for key in ("name", "@value", "url", "text"):
            if raw.get(key):
                return str(raw[key]).strip()
        return ""
    if isinstance(raw, (list, tuple)):
        for item in raw:
            text = _as_text(item)
            if text:
                return text
    return ""


def _iter_product_nodes(payload: Any) -> list[dict[str, Any]]:
    """Every ``Product`` node in a JSON-LD document, including inside ``@graph``.

    Real stores nest these two or three levels deep (``@graph`` -> ``ItemPage``
    -> ``Product``), so the walk is recursive rather than a top-level lookup.
    """
    found: list[dict[str, Any]] = []
    seen: set[int] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if id(node) in seen:  # pragma: no cover - defensive against cycles
                return
            seen.add(id(node))
            node_type = node.get("@type")
            types = node_type if isinstance(node_type, list) else [node_type]
            if any(str(entry).strip().lower() == "product" for entry in types if entry):
                found.append(node)
            for value in node.values():
                if isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return found


def _first_offer(node: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The primary ``Offer`` for a product, plus any siblings.

    ``offers`` is sometimes a single object, sometimes a list (one per variant).
    The first offer that actually declares a price wins; the rest are kept so
    their URLs can be reported.
    """
    raw = node.get("offers")
    offers: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        offers = [raw]
    elif isinstance(raw, list):
        offers = [entry for entry in raw if isinstance(entry, dict)]

    primary: dict[str, Any] = {}
    for offer in offers:
        if _offer_price(offer) is not None:
            primary = offer
            break
    if not primary and offers:
        primary = offers[0]
    return primary, offers


def _offer_price(offer: dict[str, Any]) -> float | None:
    """Price from an ``Offer``, including the ``priceSpecification`` shape."""
    for key in ("price", "lowPrice", "highPrice"):
        price = _parse_money(offer.get(key))
        if price is not None:
            return price
    specification = offer.get("priceSpecification")
    candidates = (
        specification if isinstance(specification, list) else [specification]
    )
    for entry in candidates:
        if isinstance(entry, dict):
            price = _parse_money(entry.get("price"))
            if price is not None:
                return price
    return None


def _offer_url(offer: dict[str, Any]) -> str:
    for key in ("url", "@id"):
        value = offer.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


@dataclass
class _StructuredFacts:
    product_name: str = ""
    description: str = ""
    brand: str = ""
    seller: str = ""
    price_usd: float | None = None
    currency: str = ""
    in_stock: bool | None = None
    availability_raw: str = ""
    source: str = ""
    offer_urls: list[str] = field(default_factory=list)


def _facts_from_json_ld(blocks: list[str]) -> _StructuredFacts:
    """Merge every JSON-LD ``Product`` block into one fact set."""
    facts = _StructuredFacts()
    for block in blocks:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            # A broken block is not a fact; the next one may still be good.
            continue
        for node in _iter_product_nodes(payload):
            if not facts.product_name:
                facts.product_name = _as_text(node.get("name"))
            if not facts.description:
                facts.description = _as_text(node.get("description"))
            if not facts.brand:
                facts.brand = _as_text(node.get("brand"))
            offer, offers = _first_offer(node)
            if offer:
                for entry in offers:
                    url = _offer_url(entry)
                    if url and url not in facts.offer_urls:
                        facts.offer_urls.append(url)
                if facts.price_usd is None:
                    price = _offer_price(offer)
                    if price is not None:
                        facts.price_usd = price
                        # Only a structured price counts as this source winning.
                        facts.source = "json-ld"
                if not facts.currency:
                    facts.currency = _parse_currency(
                        offer.get("priceCurrency"), default=""
                    )
                if not facts.currency:
                    specification = offer.get("priceSpecification")
                    for entry in (
                        specification if isinstance(specification, list) else [specification]
                    ):
                        if isinstance(entry, dict) and entry.get("priceCurrency"):
                            facts.currency = _parse_currency(entry["priceCurrency"])
                            break
                if not facts.seller:
                    facts.seller = _as_text(offer.get("seller"))
                if facts.in_stock is None:
                    raw = offer.get("availability")
                    facts.availability_raw = _as_text(raw)
                    facts.in_stock = _parse_availability(raw)
            if not facts.seller:
                facts.seller = _as_text(node.get("seller"))
        if facts.price_usd is not None and facts.product_name:
            break
    return facts


def _facts_from_meta(meta: dict[str, str]) -> _StructuredFacts:
    """OpenGraph / meta fallback: the shape Shopify and WooCommerce both emit."""
    facts = _StructuredFacts()
    for key in ("og:title", "twitter:title"):
        if meta.get(key):
            facts.product_name = meta[key]
            break
    for key in ("og:description", "description", "twitter:description"):
        if meta.get(key):
            facts.description = meta[key]
            break
    for key in ("product:brand", "og:brand", "product:brand:name"):
        if meta.get(key):
            facts.brand = meta[key]
            break
    for key in ("product:price:amount", "og:price:amount", "product:sale_price:amount"):
        price = _parse_money(meta.get(key))
        if price is not None:
            facts.price_usd = price
            facts.source = "opengraph"
            break
    for key in ("product:price:currency", "og:price:currency"):
        if meta.get(key):
            facts.currency = _parse_currency(meta[key])
            break
    for key in ("og:availability", "product:availability"):
        if meta.get(key):
            facts.availability_raw = meta[key]
            facts.in_stock = _parse_availability(meta[key])
            break
    for key in ("product:retailer_title", "product:retailer"):
        if meta.get(key):
            facts.seller = meta[key]
            break
    return facts


def extract_product_from_html(
    html: str, *, url: str = "", fetched_status: int = 200
) -> ExtractedWebProduct:
    """Turn a fetched page into :class:`ExtractedWebProduct`.

    Structured data wins over meta tags, which win over the ``<title>``. Only
    the first two can produce a price; the title alone can never be quoted as
    one.
    """
    parser = _PageParser()
    try:
        parser.feed(html[:MAX_HTML_CHARS])
        parser.close()
    except Exception:  # pragma: no cover - HTMLParser is very forgiving
        pass

    json_ld = _facts_from_json_ld(parser.json_ld_blocks)
    meta = _facts_from_meta(parser.meta)

    # Which source actually produced a price decides provenance.
    if json_ld.price_usd is not None:
        chosen, price_source = json_ld, "json-ld"
    elif meta.price_usd is not None:
        chosen, price_source = meta, "opengraph"
    else:
        # No structured price anywhere: merge what facts exist, but the price
        # stays None and provenance drops to ESTIMATED.
        chosen = _StructuredFacts(
            product_name=json_ld.product_name or meta.product_name or parser.title,
            description=json_ld.description or meta.description,
            brand=json_ld.brand or meta.brand,
            seller=json_ld.seller or meta.seller,
            price_usd=None,
            currency=json_ld.currency or meta.currency,
            in_stock=json_ld.in_stock if json_ld.in_stock is not None else meta.in_stock,
            availability_raw=json_ld.availability_raw or meta.availability_raw,
            source="",
            offer_urls=json_ld.offer_urls,
        )
        price_source = ""

    # Fill any gap in the price-bearing source from the other one, without
    # letting a weaker source supply the price itself.
    if price_source:
        chosen.product_name = chosen.product_name or json_ld.product_name or meta.product_name or parser.title
        chosen.description = chosen.description or json_ld.description or meta.description
        chosen.brand = chosen.brand or json_ld.brand or meta.brand
        chosen.seller = chosen.seller or json_ld.seller or meta.seller
        chosen.currency = chosen.currency or json_ld.currency or meta.currency
        if chosen.in_stock is None:
            chosen.in_stock = json_ld.in_stock if json_ld.in_stock is not None else meta.in_stock
        chosen.availability_raw = chosen.availability_raw or json_ld.availability_raw or meta.availability_raw
        for extra_url in json_ld.offer_urls:
            if extra_url not in chosen.offer_urls:
                chosen.offer_urls.append(extra_url)

    if price_source:
        provenance = DataProvenance.VERIFIED
        escalation = ""
    elif chosen.product_name or chosen.description:
        provenance = DataProvenance.ESTIMATED
        escalation = (
            "Page carried no structured price (no JSON-LD Product/Offer and no "
            "product:price:amount meta tag), so no price was reported. A product "
            "page must declare its price in structured data before it can be quoted "
            "as a verified cost."
        )
    else:
        provenance = DataProvenance.ESTIMATED
        escalation = (
            "Page yielded no product name, description or structured price - it is "
            "probably not a product page. Nothing was extracted."
        )

    return ExtractedWebProduct(
        url=url,
        product_name=chosen.product_name.strip(),
        description=chosen.description.strip(),
        brand=chosen.brand.strip(),
        seller=chosen.seller.strip(),
        price_usd=chosen.price_usd,
        currency=chosen.currency,
        in_stock=chosen.in_stock,
        availability_raw=chosen.availability_raw.strip(),
        provenance=provenance,
        # The receipt: which URL the fact came from. Empty for a page that
        # produced no structured price, because there is nothing verified to
        # attribute.
        source_reference=url if price_source else "",
        extraction_source=price_source or "meta-only",
        offer_urls=tuple(chosen.offer_urls),
        escalation_reason=escalation,
        extra={
            "http_status": fetched_status,
            "json_ld_blocks": len(parser.json_ld_blocks),
            "meta_tags": len(parser.meta),
        },
    )


# ══════════════════════════════════════════════════════════════════════════
# Service
# ══════════════════════════════════════════════════════════════════════════


class WebResearchService:
    """Fetch product pages and search the web, with injectable transport.

    ``transport`` exists for the same reason it does on
    :class:`~app.core.ai_provider.OpenAICompatibleProvider`: the whole HTTP +
    parsing pipeline is exercised by ``httpx.MockTransport`` with no network,
    no flakiness and no reliance on a real merchant staying online.

    ``resolver`` is the SSRF escape hatch - tests pass ``lambda host: []``-style
    stubs so a fake domain like ``shop.example.com`` can be validated without a
    DNS lookup. Production uses :func:`socket.getaddrinfo`.
    """

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float | None = None,
        client: httpx.Client | None = None,
        resolver: Any = None,
        search_provider: str | None = None,
        search_api_key: str | None = None,
        search_endpoint: str | None = None,
    ) -> None:
        self._timeout = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else float(settings.web_research_timeout_seconds)
        )
        self._owns_client = client is None
        self._client = client or httpx.Client(
            headers=BROWSER_HEADERS,
            timeout=self._timeout,
            transport=transport,
            follow_redirects=True,
        )
        self._resolver = resolver
        self.search_provider = (
            search_provider if search_provider is not None else settings.web_search_provider
        ).strip().lower()
        self._search_api_key = (
            search_api_key
            if search_api_key is not None
            else settings.web_search_api_key
        )
        self._search_endpoint = (
            search_endpoint
            if search_endpoint is not None
            else settings.web_search_endpoint
        )
        if self.search_provider not in SEARCH_PROVIDERS:
            raise ValueError(
                f"WEB_SEARCH_PROVIDER must be one of {', '.join(SEARCH_PROVIDERS)} "
                f"(got {self.search_provider!r})"
            )

    # ── lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the owned ``httpx.Client``. Idempotent; injected clients stay open."""
        if self._owns_client:
            self._client.close()
            self._owns_client = False

    def __enter__(self) -> WebResearchService:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── SSRF-aware validation ─────────────────────────────────────────────

    def _validate(self, url: str) -> str:
        """Scheme/host validation, with the injectable resolver for tests."""
        validated = validate_public_url(url) if self._resolver is None else self._validate_with_resolver(url)
        return validated

    def _validate_with_resolver(self, url: str) -> str:
        """Same rules as :func:`validate_public_url`, minus the real DNS call."""
        text = (url or "").strip()
        parts = urlsplit(text)
        scheme = (parts.scheme or "").lower()
        if scheme not in ALLOWED_SCHEMES:
            raise UnsafeUrlError(
                f"Refusing to fetch scheme {scheme or '(none)'!r}: only http and https are allowed"
            )
        hostname = (parts.hostname or "").strip().lower().rstrip(".")
        if not hostname:
            raise UnsafeUrlError(f"URL {text!r} has no hostname")
        if hostname in BLOCKED_HOSTNAMES:
            raise UnsafeUrlError(
                f"Refusing to fetch {hostname!r}: loopback, local and cloud-metadata "
                "hostnames are blocked"
            )
        if any(hostname.endswith(suffix) for suffix in BLOCKED_HOST_SUFFIXES):
            raise UnsafeUrlError(
                f"Refusing to fetch {hostname!r}: private-network hostnames are blocked"
            )
        try:
            literal = ipaddress.ip_address(hostname.strip("[]"))
        except ValueError:
            literal = None
        if literal is not None:
            if not _ip_is_public(literal):
                raise UnsafeUrlError(
                    f"Refusing to fetch {hostname}: private, loopback, link-local and "
                    "reserved addresses are blocked"
                )
            return text
        try:
            resolved_addresses = list(self._resolver(hostname) or [])
        except Exception as exc:  # resolver failures are refusals, not crashes
            raise UnsafeUrlError(
                f"Refusing to fetch {hostname!r}: hostname did not resolve ({exc})"
            ) from exc
        for address in resolved_addresses:
            try:
                resolved = ipaddress.ip_address(str(address).split("%", 1)[0])
            except ValueError:
                raise UnsafeUrlError(
                    f"Refusing to fetch {hostname!r}: unparseable address {address!r}"
                ) from None
            if not _ip_is_public(resolved):
                raise UnsafeUrlError(
                    f"Refusing to fetch {hostname!r}: it resolves to the non-public "
                    f"address {resolved}"
                )
        return text

    # ── page fetch ────────────────────────────────────────────────────────

    def fetch_product_page(self, url: str) -> ExtractedWebProduct:
        """Fetch one product page and extract its structured facts.

        Raises :class:`UnsafeUrlError` for a blocked URL (before any request)
        and :class:`WebResearchError` for a transport/HTTP failure. A page that
        simply lacks structured data is *not* an error: it returns an
        ``ExtractedWebProduct`` with ``price_usd = None``.
        """
        target = self._validate(url)
        response = self._get(target)
        html = _decode_body(response)
        return extract_product_from_html(html, url=target, fetched_status=response.status_code)

    def fetch_product_pages(
        self, urls: list[str], *, limit: int = 10
    ) -> tuple[list[ExtractedWebProduct], list[dict[str, Any]]]:
        """Fetch several pages, collecting per-URL failures instead of raising.

        One dead link must not lose the data from the other nine, so failures are
        returned alongside the successes and every one of them says why.
        """
        products: list[ExtractedWebProduct] = []
        errors: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_url in urls or []:
            text = str(raw_url).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            if len(products) >= limit:
                errors.append(
                    {
                        "url": text,
                        "error": f"skipped: per-call limit of {limit} pages reached",
                    }
                )
                continue
            try:
                products.append(self.fetch_product_page(text))
            except UnsafeUrlError as exc:
                errors.append({"url": text, "error": str(exc), "blocked": True})
            except WebResearchError as exc:
                errors.append({"url": text, "error": str(exc), "blocked": False})
        return products, errors

    def _get(self, url: str) -> httpx.Response:
        try:
            response = self._client.get(url, headers={"Accept-Encoding": "identity"})
        except httpx.HTTPError as exc:
            raise WebResearchError(f"Request to {url} failed: {exc}") from exc
        if response.status_code >= 400:
            raise WebResearchError(
                f"{url} returned HTTP {response.status_code}"
            )
        return response

    # ── search ────────────────────────────────────────────────────────────

    def search_web_products(
        self, query: str, max_results: int = DEFAULT_SEARCH_RESULTS
    ) -> list[WebSearchHit]:
        """Return candidate product URLs for a query.

        With ``WEB_SEARCH_PROVIDER=none`` (the default) no network call is made
        at all and the result is empty - callers then fall back to the offline
        discovery adapter instead of silently getting fabricated URLs.
        """
        text = (query or "").strip()
        if not text:
            return []
        limit = max(1, min(int(max_results), MAX_SEARCH_RESULTS))
        if self.search_provider == "none":
            return []
        if self.search_provider == "duckduckgo":
            return self._search_duckduckgo(text, limit)
        return self._search_custom(text, limit)

    def _search_duckduckgo(self, query: str, limit: int) -> list[WebSearchHit]:
        """Parse the keyless DuckDuckGo HTML endpoint.

        Only the public, robots-friendly HTML endpoint is used, and every hit is
        SSRF-validated exactly like a caller-supplied URL - a search result is
        still an untrusted URL.
        """
        url = f"{DUCKDUCKGO_HTML_ENDPOINT}?q={quote_plus(query)}"
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise WebResearchError(f"Search request failed: {exc}") from exc
        if response.status_code >= 400:
            raise WebResearchError(
                f"Search endpoint returned HTTP {response.status_code}"
            )
        return _parse_duckduckgo_html(response.text)[:limit]

    def _search_custom(self, query: str, limit: int) -> list[WebSearchHit]:
        """Call a configured JSON search API (``WEB_SEARCH_ENDPOINT``)."""
        if not self._search_endpoint:
            raise WebResearchError(
                "WEB_SEARCH_PROVIDER=custom requires WEB_SEARCH_ENDPOINT"
            )
        headers = {"Accept": "application/json"}
        if self._search_api_key:
            headers["Authorization"] = f"Bearer {self._search_api_key}"
        try:
            response = self._client.get(
                self._search_endpoint,
                params={"q": query, "limit": limit},
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise WebResearchError(f"Search request failed: {exc}") from exc
        if response.status_code >= 400:
            raise WebResearchError(
                f"Search endpoint returned HTTP {response.status_code}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise WebResearchError(f"Search endpoint returned non-JSON: {exc}") from exc
        return _parse_search_json(payload)[:limit]


def _decode_body(response: httpx.Response) -> str:
    """Decode the first slice of a response body, honouring the header charset."""
    body = response.content[:MAX_RESPONSE_BYTES]
    encoding = response.encoding or "utf-8"
    try:
        return body.decode(encoding, errors="replace")
    except (LookupError, UnicodeDecodeError):  # pragma: no cover - odd charsets
        return body.decode("utf-8", errors="replace")


#: DuckDuckGo HTML result anchors.
_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_DDG_SNIPPET_RE = re.compile(
    r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(raw: str) -> str:
    """Strip tags *and* entities: search results are HTML-escaped text."""
    return re.sub(r"\s+", " ", html_module.unescape(_TAG_RE.sub(" ", raw or ""))).strip()


def _unwrap_redirect(href: str) -> str:
    """DuckDuckGo wraps results in ``/l/?uddg=<encoded target>``."""
    if not href:
        return ""
    parts = urlsplit(href)
    if parts.path.startswith("/l/"):
        target = parse_qs(parts.query).get("uddg")
        if target:
            return target[0]
    if href.startswith("//"):
        return f"https:{href}"
    return href


def _parse_duckduckgo_html(html: str) -> list[WebSearchHit]:
    """Extract results, validating every URL the same way as a caller's URL."""
    hits: list[WebSearchHit] = []
    snippets = [_strip_tags(match.group("snippet")) for match in _DDG_SNIPPET_RE.finditer(html)]
    for index, match in enumerate(_DDG_RESULT_RE.finditer(html)):
        url = _unwrap_redirect(match.group("href").strip())
        if not url:
            continue
        parts = urlsplit(url)
        if (parts.scheme or "").lower() not in _BLOCKED_RESULT_SCHEMES:
            continue
        hits.append(
            WebSearchHit(
                url=url,
                title=_strip_tags(match.group("title")),
                snippet=snippets[index] if index < len(snippets) else "",
                source="duckduckgo",
                rank=index + 1,
            )
        )
    return hits


def _parse_search_json(payload: Any) -> list[WebSearchHit]:
    """Accept the common ``{"results": [{"url", "title", "snippet"}]}`` shape."""
    entries: Any = payload
    if isinstance(payload, dict):
        for key in ("results", "organic_results", "items", "data", "web"):
            if isinstance(payload.get(key), (list, dict)):
                entries = payload[key]
                break
    if isinstance(entries, dict):
        entries = entries.get("results") or entries.get("value") or []
    hits: list[WebSearchHit] = []
    for index, entry in enumerate(entries if isinstance(entries, list) else []):
        if isinstance(entry, str):
            url, title, snippet = entry, "", ""
        elif isinstance(entry, dict):
            url = str(
                entry.get("url")
                or entry.get("link")
                or entry.get("href")
                or entry.get("Url")
                or ""
            )
            title = str(entry.get("title") or entry.get("name") or entry.get("Title") or "")
            snippet = str(
                entry.get("snippet") or entry.get("description") or entry.get("body") or ""
            )
        else:
            continue
        url = url.strip()
        parts = urlsplit(url)
        if not url or (parts.scheme or "").lower() not in _BLOCKED_RESULT_SCHEMES:
            continue
        hits.append(
            WebSearchHit(
                url=url,
                title=title.strip(),
                snippet=snippet.strip(),
                source="custom",
                rank=index + 1,
            )
        )
    return hits


def default_search_service() -> WebResearchService:
    """A service configured entirely from :data:`app.core.config.settings`."""
    return WebResearchService()
