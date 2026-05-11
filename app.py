import os

import anthropic


MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


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
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return "\n".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )


def main():
    destination = input("Where would you like to travel? ").strip()

    if not destination:
        print("Please enter a destination so I can build an itinerary.")
        return

    trip_length = input("Trip length (3 days, 5 days, 1 week) [3 days]: ").strip() or "3 days"
    if trip_length not in {"3 days", "5 days", "1 week"}:
        print("Using 3 days because the trip length was not recognized.")
        trip_length = "3 days"

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set the ANTHROPIC_API_KEY environment variable and try again.")
        return

    try:
        itinerary = build_itinerary(destination, trip_length)
    except anthropic.APIError as exc:
        print(f"Anthropic API error: {exc}")
        return
    except Exception as exc:
        print(f"Could not generate an itinerary: {exc}")
        return

    print(f"\nAI-generated {trip_length} itinerary for {destination}")
    print("-" * (29 + len(trip_length) + len(destination)))
    print(itinerary)


if __name__ == "__main__":
    main()
