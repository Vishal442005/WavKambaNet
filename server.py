"""
NeuroDecode BCI Web Application Server
======================================
Serves the interactive BCI GUI on http://localhost:8080.
"""

import sys
import os
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

PORT = 8080
DIRECTORY = Path(__file__).resolve().parent

class BCIRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

def run_server(port=PORT):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, BCIRequestHandler)
    print("=" * 70)
    print("NeuroDecode BCI Web Application Live Server")
    print(f"Local Access URL: http://localhost:{port}/index.html")
    print(f"Serving from: {DIRECTORY}")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping BCI Server...")
        httpd.server_close()

if __name__ == "__main__":
    p = PORT
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        p = int(sys.argv[1])
    run_server(p)
