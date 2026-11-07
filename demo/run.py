#!/usr/bin/env python
"""One-command launcher for the Agent Console.

Usage
-----
    python demo/run.py                  # http://127.0.0.1:8000
    python demo/run.py --port 9000
    python demo/run.py --no-browser     # do not try to open a browser tab

Why this exists
---------------
The console needs three environment variables and a port, and getting one of
them wrong produces a confusing failure ("no such table", or a preview that
shows nothing). This script sets sane defaults, creates the SQLite file, and
reports the exact URL to open - so running the demo is one command with no
setup.

Nothing here changes how the app behaves: it only chooses defaults and then
launches the same ``uvicorn demo.app:app`` you would type by hand.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Running `python demo/run.py` puts `demo/` on sys.path, not the repo root, so
# uvicorn's "demo.app:app" import string would fail. Put the root on the path
# first: the launcher must work from any working directory.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: The demo database lives beside the code, not beside whatever directory you
#: happened to run the command from - otherwise a run from elsewhere would
#: quietly create a second, empty database.
DEMO_DB = REPO_ROOT / "demo_console.db"

#: Defaults for a local demo run. ``AI_PROVIDER=mock`` keeps the demo free and
#: deterministic; ``WEB_SEARCH_PROVIDER=none`` keeps every request on this
#: machine, since the console only ever reads its bundled sample pages.
DEFAULTS = {
    "APP_ENV": "development",
    "DATABASE_URL": f"sqlite:///{DEMO_DB.as_posix()}",
    "AI_PROVIDER": "mock",
    "WEB_SEARCH_PROVIDER": "none",
    "SQL_ECHO": "false",
}


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _open_when_ready(url: str, port: int, timeout: float = 20.0) -> None:
    """Open a browser once the server actually accepts connections.

    Opening immediately usually wins the race and shows "connection refused",
    so wait for the port and let the browser see a live server.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.4)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                webbrowser.open(url)
                return
        time.sleep(0.25)
    print(f"(server did not answer on port {port} within {timeout:.0f}s)", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Dropshipping AI Agent Console.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="port (default 8000)")
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser tab"
    )
    parser.add_argument(
        "--reload", action="store_true", help="auto-reload on code changes (dev)"
    )
    args = parser.parse_args(argv)

    # Settings are read at import time, so the environment must be set before
    # anything under `app` or `demo` is imported.
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)

    if not _port_is_free(args.port, "127.0.0.1") and args.host in {"127.0.0.1", "localhost"}:
        print(
            f"Port {args.port} is already in use on this machine.\n"
            f"Either open http://127.0.0.1:{args.port} (it may already be running),\n"
            f"or start on another port:  python demo/run.py --port {args.port + 1}",
            file=sys.stderr,
        )
        return 1

    # 0.0.0.0 so the console is reachable from a preview proxy or another device;
    # the URL to open is still written in terms of the host you asked for.
    display_host = "127.0.0.1" if args.host in {"0.0.0.0", ""} else args.host
    url = f"http://{display_host}:{args.port}/"

    print("Dropshipping AI System - Agent Console")
    print(f"  environment : {os.environ['APP_ENV']}   AI provider: {os.environ['AI_PROVIDER']}")
    print(f"  database    : {os.environ['DATABASE_URL']}")
    print(f"  UI          : {url}")
    print(f"  API docs    : {url}docs")
    print("  sample pages: 4 bundled pages under demo-shop.example (no internet needed)")
    print("  stop with   : Ctrl+C\n")

    if not args.no_browser:
        threading.Thread(
            target=_open_when_ready, args=(url, args.port), daemon=True
        ).start()

    import uvicorn

    try:
        uvicorn.run(
            "demo.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            access_log=False,
        )
    except KeyboardInterrupt:  # Ctrl+C is the documented way to stop
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
