import json
import os
import sys
import io
import re
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import anthropic


HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8000"))
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
BASE_DIR = Path(__file__).resolve().parent


def configure_text_encoding():
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
        elif stream and hasattr(stream, "buffer"):
            setattr(
                sys,
                stream_name,
                io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace"),
            )


def safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        print(str(message).encode("ascii", errors="backslashreplace").decode("ascii"))


def build_prompt(message):
    return f"""
A traveler just sent you this message: "{message}"

You're their well-travelled friend. Reply with a punchy, personalized travel plan.

Your FIRST line must be exactly this format (no extra text):
DESTINATION: [the place name or best guess — e.g. "Hudson, WI" or "Hawaii" or "Amalfi Coast"]

Then write the travel plan. Rules:
- Casual, direct, fun — like texting a friend who's been there.
- Short sentences. Fragments are fine. No travel brochure language.
- If they gave a trip length, use it. Otherwise default to 3 days.
- Day-by-day plan (Day 1, Day 2, etc.) — one or two punchy sentences each.
- "Where to stay:" — one budget pick, one splurge, with rough nightly price.
- "Don't miss:" — one single must-do thing.
- Address any specific requests they made (group size, vibe, occasion, budget).
- Total under 220 words.
""".strip()


def stream_itinerary_words(message):
    client = anthropic.Anthropic()
    pending = ""

    with client.messages.stream(
        model=MODEL,
        max_tokens=2500,
        system=(
            "You are a well-travelled friend giving honest, casual travel advice. "
            "Short sentences. Real recommendations. No filler."
        ),
        messages=[{"role": "user", "content": build_prompt(message)}],
    ) as stream:
        for text in stream.text_stream:
            pending += text
            last_whitespace = max(pending.rfind(char) for char in (" ", "\n", "\r", "\t"))

            if last_whitespace == -1:
                continue

            ready = pending[: last_whitespace + 1]
            pending = pending[last_whitespace + 1 :]

            for word in re.findall(r"\s+|\S+\s*", ready):
                yield word

    if pending:
        yield pending


class TravelRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)

        if parsed_url.path in ("/", "/index.html"):
            self.serve_index()
            return

        if parsed_url.path == "/api/itinerary/stream":
            self.stream_itinerary(parsed_url.query)
            return

        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def stream_itinerary(self, query_string):
        if not os.getenv("ANTHROPIC_API_KEY"):
            self.send_sse_error(
                "Set the ANTHROPIC_API_KEY environment variable first.",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        query = parse_qs(query_string)
        message = str((query.get("message") or [""])[0]).strip()

        if not message:
            self.send_sse_error("Message is required.", HTTPStatus.BAD_REQUEST)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            self.write_sse("start", {"message": message})
            for word in stream_itinerary_words(message):
                self.write_sse("token", {"text": word})
            self.write_sse("done", {"message": message})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            traceback.print_exc()
            try:
                self.write_sse("error", {"error": f"Could not generate an itinerary: {exc}"})
            except (BrokenPipeError, ConnectionResetError):
                return

    def do_POST(self):
        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def serve_index(self):
        index_path = BASE_DIR / "index.html"
        try:
            content = index_path.read_bytes()
        except OSError:
            self.send_json({"error": "index.html not found"}, HTTPStatus.NOT_FOUND)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload, status=HTTPStatus.OK):
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_sse_error(self, message, status=HTTPStatus.OK):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.write_sse("error", {"error": message})

    def write_sse(self, event, payload):
        self.wfile.write(f"event: {event}\n".encode("utf-8"))
        data = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def log_message(self, format, *args):
        safe_print("%s - - %s" % (self.address_string(), format % args))


def main():
    configure_text_encoding()
    server = ThreadingHTTPServer((HOST, PORT), TravelRequestHandler)
    safe_print(f"Travel app running at http://{HOST}:{PORT}")
    safe_print("Press Ctrl+C to stop the server.")
    server.serve_forever()


if __name__ == "__main__":
    main()
