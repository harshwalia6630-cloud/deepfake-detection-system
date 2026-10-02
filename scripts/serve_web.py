"""Static server for web/ with the same COOP/COEP headers Vercel sends (enables threaded WASM)."""

import functools
import http.server
import sys
from pathlib import Path


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8200
    web = Path(__file__).resolve().parents[1] / "web"
    http.server.ThreadingHTTPServer(("", port), functools.partial(Handler, directory=str(web))).serve_forever()
