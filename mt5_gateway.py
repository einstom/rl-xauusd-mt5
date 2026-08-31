"""Tiny HTTP gateway exposing the MetaTrader5 package to another Python.

Run this INSIDE the Windows Python that can import MetaTrader5 — on macOS
that is the Windows Python installed into the same Wine bottle as the MT5
terminal; on a Windows box it is plain Python.  The trading side (native
macOS/Linux Python with torch/SB3) then talks to it via mt5_compat.GatewayMT5
by setting MT5_GATEWAY_URL.

Deliberately dependency-free (stdlib + MetaTrader5 + numpy, which the
MetaTrader5 package requires anyway), because installing anything heavier
under Wine is exactly the problem this gateway avoids.

    wine python.exe mt5_gateway.py --port 8787

Endpoints
---------
GET  /constants   all integer constants of the MetaTrader5 module
POST /call        {"fn": name, "args": [...], "kwargs": {...}} → {"ok", "result"}
                  datetimes travel as {"__dt__": epoch_seconds} (UTC-stamped,
                  matching how copy_rates_range expects its window arguments)

Binds 127.0.0.1 only.  Set MT5_GATEWAY_TOKEN (same value on both sides) to
additionally require an X-Gateway-Token header.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import MetaTrader5 as mt5

try:
    import numpy as np
except ImportError:  # MetaTrader5 depends on numpy, so this should not happen
    np = None

TOKEN = os.environ.get("MT5_GATEWAY_TOKEN")


def _constants() -> dict[str, int]:
    out = {}
    for name in dir(mt5):
        if name.startswith("_"):
            continue
        value = getattr(mt5, name)
        if isinstance(value, int) and not isinstance(value, bool):
            out[name] = value
    return out


def _to_jsonable(obj):
    """MT5 results (namedtuples, tuples of them, numpy rate arrays) → JSON."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "_asdict"):                       # namedtuple (SymbolInfo, …)
        return {k: _to_jsonable(v) for k, v in obj._asdict().items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if np is not None and isinstance(obj, np.ndarray):
        if obj.dtype.names:                           # structured rates array
            return [dict(zip(obj.dtype.names, (_to_jsonable(x) for x in rec.tolist())))
                    for rec in obj]
        return obj.tolist()
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):                # positions_get / orders_get
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.timestamp()
    return repr(obj)


def _decode(value):
    """Reverse of mt5_compat._encode: {"__dt__": epoch} → aware datetime."""
    if isinstance(value, dict):
        if set(value.keys()) == {"__dt__"}:
            return datetime.fromtimestamp(value["__dt__"], tz=timezone.utc)
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        return not TOKEN or self.headers.get("X-Gateway-Token") == TOKEN

    def do_GET(self):
        if not self._authorized():
            return self._send(403, {"ok": False, "error": "bad token"})
        if self.path == "/constants":
            return self._send(200, _constants())
        return self._send(404, {"ok": False, "error": "unknown path"})

    def do_POST(self):
        if not self._authorized():
            return self._send(403, {"ok": False, "error": "bad token"})
        if self.path != "/call":
            return self._send(404, {"ok": False, "error": "unknown path"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length).decode())
            fn_name = req["fn"]
            if fn_name.startswith("_"):
                raise ValueError(f"function name not allowed: {fn_name}")
            fn = getattr(mt5, fn_name, None)
            if not callable(fn):
                raise ValueError(f"MetaTrader5 has no function '{fn_name}'")
            args = _decode(req.get("args", []))
            kwargs = _decode(req.get("kwargs", {}))
            result = fn(*args, **kwargs)
            self._send(200, {"ok": True, "result": _to_jsonable(result)})
        except Exception as e:  # surface the error to the client, keep serving
            self._send(200, {"ok": False, "error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):  # quiet request log; errors go to client
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="MetaTrader5 HTTP gateway.")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"MT5 gateway listening on http://127.0.0.1:{args.port} "
          f"(token {'required' if TOKEN else 'not set'})")
    print("Point the trading side at it with: "
          f"export MT5_GATEWAY_URL=http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Gateway stopped.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
