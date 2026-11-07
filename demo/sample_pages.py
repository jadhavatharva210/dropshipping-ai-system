"""Sample product pages bundled with the demo console.

Why these exist
---------------
The extraction layer only trusts what a page declares in *machine-readable*
markup. To show that honestly without the console depending on the open
internet, three pages are shipped as fixtures, each exercising a different real
world shape the extractor supports:

``shopify``   a Shopify product page: ``@graph`` JSON-LD with a ``Product``
              whose ``offers`` carry ``price``/``priceCurrency``/``availability``.
``woocommerce`` a WooCommerce page: ``offers`` is a *list* of variants and the
              price lives in ``priceSpecification``.
``opengraph`` a theme with no JSON-LD at all: the price is only in
              ``product:price:amount``, which is the documented fallback.
``prose``     a page that talks about price in visible text only. It must yield
              **no** price - this is the anti-fabrication case.

The HTML is deliberately small but structurally faithful; it is parsed by the
real :func:`app.services.web_research.extract_product_from_html`, not by any
demo-only shortcut.
"""

from __future__ import annotations

__all__ = ["DEMO_PAGES", "DEMO_CATALOGUE", "page_html", "page_url"]


#: Host used for demo pages. ``.example`` is reserved by RFC 2606, so these URLs
#: can never collide with a real shop - and the demo transport intercepts them
#: before a request is even built.
DEMO_HOST = "demo-shop.example"

_SHIPIFY = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>GlowPaw LED Dog Collar - Rechargeable, Waterproof | BrightPet Supply</title>
<meta name="description" content="Rechargeable LED dog collar, waterproof, three light modes, USB-C.">
<meta property="og:title" content="GlowPaw LED Dog Collar">
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@graph": [
    {"@type": "WebSite", "name": "BrightPet Supply", "url": "https://demo-shop.example/"},
    {"@type": "Organization", "name": "BrightPet Supply"},
    {"@type": "BreadcrumbList", "itemListElement": [
        {"@type": "ListItem", "position": 1, "name": "Home"},
        {"@type": "ListItem", "position": 2, "name": "Dog Collars"}]},
    {
      "@type": "Product",
      "name": "GlowPaw LED Dog Collar",
      "sku": "BP-GLOW-01",
      "description": "Rechargeable LED collar with three light modes and a waterproof shell.",
      "brand": {"@type": "Brand", "name": "BrightPet"},
      "offers": {
        "@type": "Offer",
        "price": "9.40",
        "priceCurrency": "USD",
        "availability": "https://schema.org/InStock",
        "itemCondition": "https://schema.org/NewCondition"
      }
    }
  ]
}
</script>
</head>
<body><h1>GlowPaw LED Dog Collar</h1><p>$9.40</p></body></html>
"""

_WOO = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>LED Dog Collar (USB-C) - PetNova Wholesale</title>
<meta name="description" content="Wholesale LED dog collar, MOQ 50 units, USB-C charging.">
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Product",
  "name": "LED Dog Collar USB-C",
  "sku": "PN-2231",
  "offers": [
    {"@type": "Offer", "availability": "https://schema.org/OutOfStock"},
    {"@type": "Offer",
     "availability": "https://schema.org/InStock",
     "priceSpecification": {
        "@type": "UnitPriceSpecification",
        "price": "8.20",
        "priceCurrency": "USD",
        "unitText": "unit"
     }}
  ]
}
</script>
</head>
<body><h1>LED Dog Collar USB-C</h1></body></html>
"""

_OG = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>LumiCollar Night Safety Collar | TrailHound Gear</title>
<meta name="description" content="Reflective LED collar for late walks. Free shipping over $35.">
<meta property="og:title" content="LumiCollar Night Safety Collar">
<meta property="product:price:amount" content="10.75">
<meta property="product:price:currency" content="USD">
<meta property="og:availability" content="in stock">
</head>
<body><h1>LumiCollar Night Safety Collar</h1><p>Usually $14.00, now $10.75.</p></body></html>
"""

_PROSE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Why LED collars help on night walks | The Dog Blog</title>
<meta name="description" content="A short guide to visibility gear for dogs.">
</head>
<body>
<h1>Why LED collars help on night walks</h1>
<p>We compared options from $9.99 to $24.99. Prices change constantly, so check
the retailer. This article is editorial and sells nothing.</p>
</body></html>
"""

#: Fixture id -> HTML body.
DEMO_PAGES: dict[str, str] = {
    "shopify": _SHIPIFY,
    "woocommerce": _WOO,
    "opengraph": _OG,
    "prose": _PROSE,
}

#: Path used in demo URLs, matching what a product page usually looks like.
_DEMO_PATH = "/products/{slug}"

#: Everything the console offers in its "sample pages" picker.
#: ``role`` is what the console asks the agent to treat the page as; ``expect``
#: is the sentence shown next to it, so the observed result can be compared
#: against what the page actually declares.
DEMO_CATALOGUE: tuple[dict[str, object], ...] = (
    {
        "fixture": "shopify",
        "label": "BrightPet Supply (Shopify JSON-LD)",
        "url": "https://demo-shop.example/products/glowpaw-led-collar",
        "role": "supplier",
        "expect": "Reads $9.40 from JSON-LD offers.price - a VERIFIED landed cost.",
        "declares_price": True,
    },
    {
        "fixture": "woocommerce",
        "label": "PetNova Wholesale (WooCommerce, variants)",
        "url": "https://demo-shop.example/products/led-collar-usbc",
        "role": "supplier",
        "expect": "Offers is a list: the in-stock variant's priceSpecification ($8.20) wins.",
        "declares_price": True,
    },
    {
        "fixture": "opengraph",
        "label": "TrailHound Gear (OpenGraph only)",
        "url": "https://demo-shop.example/products/lumicollar",
        "role": "competitor",
        "expect": "No JSON-LD: falls back to product:price:amount ($10.75), still VERIFIED.",
        "declares_price": True,
    },
    {
        "fixture": "prose",
        "label": "A blog post (price only in prose)",
        "url": "https://demo-shop.example/blog/led-collars-night-walks",
        "role": "supplier",
        "expect": "Mentions $9.99 in text but declares no price - must yield NO price.",
        "declares_price": False,
    },
)

_PAGE_BY_URL = {str(entry["url"]): str(entry["fixture"]) for entry in DEMO_CATALOGUE}


def page_html(url: str) -> str:
    """Return the bundled HTML for a demo URL.

    Raises :class:`KeyError` for anything else, so the demo transport can never
    quietly serve the wrong page for a URL it does not recognise.
    """
    return DEMO_PAGES[_PAGE_BY_URL[url]]


def page_url(fixture: str) -> str:
    """The demo URL that serves ``fixture``."""
    for entry in DEMO_CATALOGUE:
        if entry["fixture"] == fixture:
            return str(entry["url"])
    raise KeyError(fixture)
