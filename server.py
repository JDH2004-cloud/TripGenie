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


def build_prompt(destination, trip_length="3 days"):
    return f"""
You're a well-travelled friend texting someone their {trip_length} plan for {destination}. Keep it casual, direct, and fun — like you've actually been there.

Format:
- One line per day (Day 1, Day 2, etc.) — short, punchy sentences. No fluff.
- A "Where to stay" section with 2 options: one budget, one splurge. Include rough nightly price.
- A "Don't miss" line — one single thing they absolutely have to do.

Rules:
- Write like a human, not a travel brochure.
- Short sentences. Fragments are fine.
- No bullet points inside the day descriptions — just one or two punchy sentences per day.
- Total response under 200 words.
""".strip()


def build_itinerary(destination, trip_length="3 days"):
    client = anthropic.Anthropic()

    message = client.messages.create(
        model=MODEL,
        max_tokens=2500,
        system=(
            "You are a well-travelled friend giving honest, casual travel advice. "
            "Short sentences. Real recommendations. No filler."
        ),
        messages=[{"role": "user", "content": build_prompt(destination, trip_length)}],
    )

    text_blocks = (
        str(block.text)
        for block in message.content
        if getattr(block, "type", None) == "text"
    )
    return "\n".join(text_blocks)


def stream_itinerary_words(destination, trip_length="3 days"):
    client = anthropic.Anthropic()
    pending = ""

    with client.messages.stream(
        model=MODEL,
        max_tokens=2500,
        system=(
            "You are a well-travelled friend giving honest, casual travel advice. "
            "Short sentences. Real recommendations. No filler."
        ),
        messages=[{"role": "user", "content": build_prompt(destination, trip_length)}],
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
        destination = str((query.get("destination") or [""])[0]).strip()
        trip_length = str((query.get("tripLength") or ["3 days"])[0]).strip()

        if not destination:
            self.send_sse_error("Destination is required.", HTTPStatus.BAD_REQUEST)
            return
        if trip_length not in {"3 days", "5 days", "1 week"}:
            trip_length = "3 days"

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            self.write_sse(
                "start",
                {"destination": destination, "tripLength": trip_length},
            )
            for word in stream_itinerary_words(destination, trip_length):
                self.write_sse("token", {"text": word})
            self.write_sse("done", {"destination": destination, "tripLength": trip_length})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            traceback.print_exc()
            try:
                self.write_sse("error", {"error": f"Could not generate an itinerary: {exc}"})
            except (BrokenPipeError, ConnectionResetError):
                return

    def do_POST(self):
        if self.path != "/api/itinerary":
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        if not os.getenv("ANTHROPIC_API_KEY"):
            self.send_json(
                {"error": "Set the ANTHROPIC_API_KEY environment variable first."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length) or b"{}"
            payload = json.loads(body.decode("utf-8"))
        except UnicodeDecodeError:
            self.send_json({"error": "Request body must be UTF-8 encoded."}, HTTPStatus.BAD_REQUEST)
            return
        except (ValueError, json.JSONDecodeError):
            self.send_json({"error": "Request body must be valid JSON."}, HTTPStatus.BAD_REQUEST)
            return

        destination = str(payload.get("destination", "")).strip()
        trip_length = str(payload.get("tripLength", "3 days")).strip()
        if not destination:
            self.send_json({"error": "Destination is required."}, HTTPStatus.BAD_REQUEST)
            return
        if trip_length not in {"3 days", "5 days", "1 week"}:
            trip_length = "3 days"

        try:
            itinerary = build_itinerary(destination, trip_length)
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"error": f"Could not generate an itinerary: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self.send_json({"destination": destination, "tripLength": trip_length, "itinerary": itinerary})

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
