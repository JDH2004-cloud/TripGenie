import json
import os
import sys
import io
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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


def build_itinerary(destination, trip_length="3 days"):
    client = anthropic.Anthropic()

    prompt = f"""
Create a detailed {trip_length} travel itinerary for {destination}.

Include:
- A practical {trip_length} schedule with morning, afternoon, and evening plans
- Specific kinds of neighborhoods, landmarks, food experiences, and cultural stops
- Pacing notes so the trip does not feel rushed
- Local transportation suggestions
- A short packing list section
- A short budget tips section for young budget travelers

Write the itinerary in clear, traveler-friendly Markdown.
""".strip()

    message = client.messages.create(
        model=MODEL,
        max_tokens=2500,
        system=(
            "You are an expert travel planner. Build specific, useful itineraries "
            "with realistic pacing and practical advice."
        ),
        messages=[{"role": "user", "content": prompt}],
    )

    text_blocks = (
        str(block.text)
        for block in message.content
        if getattr(block, "type", None) == "text"
    )
    return "\n".join(text_blocks)


class TravelRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.serve_index()
            return

        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

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
