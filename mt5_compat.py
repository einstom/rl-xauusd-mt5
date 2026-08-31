"""MT5 access layer: direct Windows package OR an HTTP gateway (macOS/Linux).

``load_mt5()`` returns a module-like object exposing the subset of the
``MetaTrader5`` API this project uses, resolved in this order:

  1. ``MT5_GATEWAY_URL`` env var set → :class:`GatewayMT5`, a thin JSON-RPC
     client for ``mt5_gateway.py`` running inside the Wine/Windows Python next
     to the terminal (the only way to reach MT5 from native macOS/Linux
     Python — the MetaTrader5 package itself is Windows-only).
  2. ``import MetaTrader5`` succeeds (Windows) → the real package.
  3. Neither → ``None``; callers raise a descriptive error via their
     ``_require_mt5()`` helpers.

The gateway wire format is a generic call: ``POST /call {fn, args, kwargs}``.
Results come back as JSON; dicts are wrapped in :class:`AttrDict` so client
code keeps the namedtuple-style attribute access of the real package
(``symbol_info(...).bid``, ``order_send(...).retcode`` …), while lists of bar
dicts still feed ``pd.DataFrame`` unchanged.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime
from typing import Any


class AttrDict(dict):
    """dict with attribute access, so gateway JSON mimics MT5 namedtuples."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _encode(value: Any) -> Any:
    """Encode call arguments; datetimes travel as epoch seconds."""
    if isinstance(value, datetime):
        return {"__dt__": value.timestamp()}
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    return value


class GatewayMT5:
    """MetaTrader5-module lookalike backed by mt5_gateway.py over HTTP."""

    def __init__(self, url: str, token: str | None = None, timeout: float = 30.0):
        self._url = url.rstrip("/")
        self._token = token or os.environ.get("MT5_GATEWAY_TOKEN")
        self._timeout = timeout
        for name, value in self._get("/constants").items():
            setattr(self, name, value)

    def _request(self, path: str, payload: bytes | None = None) -> Any:
        req = urllib.request.Request(self._url + path, data=payload,
                                     headers={"Content-Type": "application/json"})
        if self._token:
            req.add_header("X-Gateway-Token", self._token)
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode())

    def _get(self, path: str) -> Any:
        return self._request(path)

    def _rpc(self, fn: str, *args: Any, **kwargs: Any) -> Any:
        payload = json.dumps({"fn": fn, "args": _encode(list(args)),
                              "kwargs": _encode(kwargs)}).encode()
        out = self._request("/call", payload)
        if not out.get("ok"):
            raise RuntimeError(f"MT5 gateway call {fn} failed: {out.get('error')}")
        return _wrap(out.get("result"))

    def __getattr__(self, name: str):
        # Constants were materialised in __init__; anything else is a remote call.
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: self._rpc(name, *args, **kwargs)


_HELP = (
    "No MT5 backend available. Either run on the Windows machine hosting the "
    "terminal (pip install MetaTrader5), or start mt5_gateway.py inside the "
    "terminal's Wine/Windows Python (macOS) and set MT5_GATEWAY_URL, e.g. "
    "MT5_GATEWAY_URL=http://127.0.0.1:8787"
)


def load_mt5():
    """Resolve the MT5 backend (gateway → real package → None)."""
    url = os.environ.get("MT5_GATEWAY_URL")
    if url:
        return GatewayMT5(url)
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError:
        return None
