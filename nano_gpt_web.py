"""Serve a lightweight browser chat UI for a trained SpiritAI checkpoint.

Run:
    python nano_gpt_web.py
Then open http://127.0.0.1:8000 in a browser.
"""

from __future__ import annotations

from http import HTTPStatus
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from spirit_inference import SpiritGenerator, trim_prompt

HOST = os.getenv("WEB_HOST", "127.0.0.1")
PORT = int(os.getenv("WEB_PORT", "8000"))
MAX_REQUEST_BYTES = int(os.getenv("WEB_MAX_REQUEST_BYTES", "65536"))

PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SpiritAI Web Chat</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #8b5cf6;
      --accent-strong: #7c3aed;
      --danger: #fca5a5;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, rgba(139, 92, 246, 0.22), transparent 35%), var(--bg);
      color: var(--text);
    }
    main {
      width: min(960px, 100%);
      min-height: 100vh;
      margin: 0 auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
    }
    h1 { margin: 0; font-size: clamp(1.8rem, 4vw, 3rem); }
    .subtle { color: var(--muted); margin: 6px 0 0; }
    .status {
      padding: 8px 12px;
      border: 1px solid rgba(148, 163, 184, 0.25);
      border-radius: 999px;
      color: var(--muted);
      white-space: nowrap;
      background: rgba(15, 23, 42, 0.72);
    }
    #chat {
      flex: 1;
      min-height: 420px;
      overflow-y: auto;
      padding: 20px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 24px;
      background: rgba(17, 24, 39, 0.86);
      box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
    }
    .message {
      max-width: 80%;
      margin-bottom: 14px;
      padding: 12px 14px;
      border-radius: 16px;
      line-height: 1.5;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .message.user {
      margin-left: auto;
      background: var(--accent);
      color: white;
      border-bottom-right-radius: 4px;
    }
    .message.ai {
      margin-right: auto;
      background: var(--panel-2);
      border-bottom-left-radius: 4px;
    }
    .message.error { color: var(--danger); }
    form {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      padding: 12px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 22px;
      background: rgba(17, 24, 39, 0.92);
    }
    textarea {
      width: 100%;
      min-height: 54px;
      max-height: 180px;
      resize: vertical;
      border: 0;
      outline: 0;
      padding: 14px;
      border-radius: 16px;
      color: var(--text);
      background: #0b1220;
      font: inherit;
    }
    button {
      border: 0;
      border-radius: 16px;
      padding: 0 22px;
      color: white;
      background: var(--accent);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-strong); }
    button:disabled { cursor: wait; opacity: 0.62; }
    @media (max-width: 640px) {
      main { padding: 14px; }
      header { align-items: start; flex-direction: column; }
      form { grid-template-columns: 1fr; }
      button { min-height: 48px; }
      .message { max-width: 94%; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>SpiritAI Web Chat</h1>
        <p class="subtle">A separate, minimal browser UI for your local Nano-GPT checkpoint.</p>
      </div>
      <div id="status" class="status">Ready</div>
    </header>

    <section id="chat" aria-live="polite">
      <div class="message ai">Ask SpiritAI a question to begin.</div>
    </section>

    <form id="chat-form">
      <textarea id="prompt" name="prompt" placeholder="Type a message..." required autofocus></textarea>
      <button id="send" type="submit">Send</button>
    </form>
  </main>

  <script>
    const chat = document.querySelector('#chat');
    const form = document.querySelector('#chat-form');
    const promptInput = document.querySelector('#prompt');
    const sendButton = document.querySelector('#send');
    const status = document.querySelector('#status');

    function addMessage(text, role, extraClass = '') {
      const message = document.createElement('div');
      message.className = `message ${role} ${extraClass}`.trim();
      message.textContent = text;
      chat.appendChild(message);
      chat.scrollTop = chat.scrollHeight;
      return message;
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const prompt = promptInput.value.trim();
      if (!prompt) return;

      addMessage(prompt, 'user');
      promptInput.value = '';
      sendButton.disabled = true;
      status.textContent = 'Generating...';
      const pending = addMessage('Thinking...', 'ai');

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt }),
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || 'The server returned an error.');
        }
        pending.textContent = payload.reply || '[Empty response]';
        status.textContent = 'Ready';
      } catch (error) {
        pending.textContent = error.message;
        pending.classList.add('error');
        status.textContent = 'Error';
      } finally {
        sendButton.disabled = false;
        promptInput.focus();
      }
    });
  </script>
</body>
</html>
"""


generator = SpiritGenerator()


class ChatHandler(BaseHTTPRequestHandler):
    """HTTP routes for the standalone web chat UI."""

    server_version = "SpiritAIWeb/1.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._send_text(PAGE_HTML, "text/html; charset=utf-8")
            return
        if path == "/api/health":
            info = generator.info
            self._send_json(
                {
                    "status": "ok",
                    "device": info.device,
                    "block_size": info.block_size,
                    "vocab_size": info.vocab_size,
                }
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._send_json(
                {"error": "Request body is empty or too large."}, HTTPStatus.BAD_REQUEST
            )
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Request body must be valid JSON."}, HTTPStatus.BAD_REQUEST)
            return

        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            self._send_json({"error": "Prompt is required."}, HTTPStatus.BAD_REQUEST)
            return

        generated = generator.generate(prompt)
        self._send_json({"reply": trim_prompt(generated, prompt), "raw": generated})

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _send_json(
        self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(
        self, text: str, content_type: str, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    info = generator.info
    settings = generator.settings
    print(
        f"Loaded SpiritAI on {info.device}: layers={info.n_layer}, heads={info.n_head}, "
        f"embd={info.n_embd}, block={info.block_size}, vocab={info.vocab_size}"
    )
    print(
        "Sampling: "
        f"max_new_tokens={settings.max_new_tokens}, top_k={settings.top_k}, "
        f"temperature={settings.temperature}, repetition_penalty={settings.repetition_penalty}"
    )
    print(f"Serving SpiritAI web chat at http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), ChatHandler).serve_forever()


if __name__ == "__main__":
    main()
