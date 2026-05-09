"""Standalone Web Server for SpiritAI.

Provides a simple dark-mode UI and a REST API for generation and status
using only the Python standard library's http.server.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse

import torch

from spirit.config import CHECKPOINT_PATH
from spirit.inference.generator import SpiritGenerator

logger = logging.getLogger("spirit.web")

# Global generator instance initialized on startup
generator: SpiritGenerator | None = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SpiritAI Chat</title>
    <style>
        body { font-family: system-ui, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        #status-panel { background: #1e1e1e; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-size: 0.9em; }
        #chat-box { height: 400px; background: #1e1e1e; border-radius: 8px; padding: 20px; overflow-y: auto; margin-bottom: 20px; }
        .message { margin-bottom: 15px; }
        .user { color: #90caf9; }
        .ai { color: #a5d6a7; }
        .prompt-container { display: flex; gap: 10px; }
        input[type="text"] { flex-grow: 1; padding: 10px; border-radius: 4px; border: 1px solid #333; background: #2c2c2c; color: white; }
        button { padding: 10px 20px; border-radius: 4px; border: none; background: #1976d2; color: white; cursor: pointer; }
        button:hover { background: #1565c0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>SpiritAI</h1>
        <div id="status-panel">Loading status...</div>
        <div id="chat-box"></div>
        <div class="prompt-container">
            <input type="text" id="prompt" placeholder="Ask a theological question..." onkeypress="if(event.key === 'Enter') send()">
            <button onclick="send()">Send</button>
        </div>
    </div>
    <script>
        async function loadStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('status-panel').innerHTML =
                    `<b>Device:</b> ${data.device} | <b>Params:</b> ${data.params_millions}M | ` +
                    `<b>Layers:</b> ${data.layers} | <b>Heads:</b> ${data.heads} | ` +
                    `<b>Embed:</b> ${data.embd} | <b>Step:</b> ${data.step}`;
            } catch (e) {
                document.getElementById('status-panel').innerText = "Status unavailable.";
            }
        }

        async function send() {
            const input = document.getElementById('prompt');
            const text = input.value.trim();
            if(!text) return;

            const chatBox = document.getElementById('chat-box');
            chatBox.innerHTML += `<div class="message"><b class="user">You:</b> ${text}</div>`;
            input.value = '';

            try {
                const res = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt: text})
                });
                const data = await res.json();
                chatBox.innerHTML += `<div class="message"><b class="ai">AI:</b> ${data.response}</div>`;
            } catch (e) {
                chatBox.innerHTML += `<div class="message" style="color:red;">Error: Could not reach server.</div>`;
            }
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        loadStatus();
    </script>
</body>
</html>"""


class SpiritRequestHandler(BaseHTTPRequestHandler):
    """Handler for SpiritAI web requests."""

    def _set_headers(self, content_type: str = "application/json") -> None:
        self.send_response(200)
        self.send_header("Content-type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self._set_headers()

    def do_GET(self) -> None:
        parsed_path = urllib.parse.urlparse(self.path)

        if parsed_path.path == "/":
            self._set_headers("text/html")
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif parsed_path.path == "/api/status":
            self._set_headers()
            if generator is None:
                self.wfile.write(json.dumps({"error": "Model not loaded"}).encode("utf-8"))
                return

            cfg = generator.model_config
            # rough parameter count estimation
            params = cfg.n_layer * (12 * cfg.n_embd**2 + 13 * cfg.n_embd) + cfg.vocab_size * cfg.n_embd

            status = {
                "device": generator.device,
                "layers": cfg.n_layer,
                "heads": cfg.n_head,
                "embd": cfg.n_embd,
                "block_size": cfg.block_size,
                "vocab_size": cfg.vocab_size,
                "params_millions": round(params / 1e6, 1),
                "step": "Unknown"
            }

            # Try to grab step from checkpoint
            if CHECKPOINT_PATH.exists():
                ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
                status["step"] = ckpt.get("iter_num", 0)

            self.wfile.write(json.dumps(status).encode("utf-8"))

        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed_path = urllib.parse.urlparse(self.path)

        if parsed_path.path == "/api/generate":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode("utf-8"))
                prompt = data.get("prompt", "")

                if not prompt or generator is None:
                    raise ValueError("Invalid prompt or model not loaded")

                response = generator.generate(prompt)

                self._set_headers()
                self.wfile.write(json.dumps({"response": response}).encode("utf-8"))

            except ValueError as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            except OSError as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Internal System Error"}).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")


def start_server(port: int = 8000) -> None:
    """Initialize model and start the web server."""
    global generator
    try:
        generator = SpiritGenerator()
        logger.info(f"Model loaded on {generator.device}")
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("Starting server without model. /api/generate will fail.")

    server_address = ('', port)
    httpd = HTTPServer(server_address, SpiritRequestHandler)
    logger.info(f"Starting server on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        logger.info("Server stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    start_server(args.port)
