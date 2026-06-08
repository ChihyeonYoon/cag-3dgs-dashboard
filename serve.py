#!/usr/bin/env python3
import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000
Handler = http.server.SimpleHTTPRequestHandler

# Change directory to where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class MyHandler(Handler):
    # Enable CORS and disable caching
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

socketserver.TCPServer.allow_reuse_address = True

print(f"Starting local server at http://localhost:{PORT}...")
webbrowser.open(f"http://localhost:{PORT}")

try:
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print(f"Server is running. Press Ctrl+C to stop.")
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped. Exiting.")
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
