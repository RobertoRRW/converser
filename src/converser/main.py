import os
import json
import argparse
import asyncio
from datetime import datetime
from typing import Any

from converser import graph as graph


def parse_history(history: list[Any]) -> str:
    messages = []
    for elem in history:
        match elem:
            case {"role": "user", "content": content} as message:
                messages.append(message)
            case {"role": "assistant", "content": [{"text": content}]}:
                messages.append({"role": "assistant", "content": content})
    return json.dumps(messages)


def save_conversation(history, directory=None):
    """
    Save conversation history as JSON in the specified directory with timestamp.
    Only saves if directory is explicitly specified.
    """
    if directory is None:
        print("No directory specified, conversation not saved.")
        return None

    os.makedirs(directory, exist_ok=True)

    json_data = parse_history(history)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    filename = f"conversation-{timestamp}.json"
    filepath = os.path.join(directory, filename)

    with open(filepath, "w") as f:
        f.write(json_data)

    print(f"Conversation saved to {filepath}")
    return filepath


def parse_args():
    parser = argparse.ArgumentParser(description="Talk to a customer service agent.")
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=None,
        help="Directory to save conversation files (required for saving)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    final_output = asyncio.run(graph.run())

    if args.directory:
        save_conversation(final_output.message_history, args.directory)
    else:
        print("No directory specified. Use --directory to save the conversation.")
