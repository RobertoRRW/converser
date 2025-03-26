import csv
from dataclasses import asdict
import os
import json
import argparse
import asyncio
from datetime import datetime
from typing import Any

from converser import graph as graph
from converser.agent import ConversationState


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


def save_state_as_tsv(state: ConversationState, tsv_file=None):
    """
    Save conversation state as a TSV file.
    Appends to the file if it exists, creates if it doesn't.
    """
    if tsv_file is None:
        print("No TSV file specified, state not saved.")
        return None

    state_dict = asdict(state)

    state_dict["attempted_solutions"] = json.dumps(state_dict["attempted_solutions"])
    state_dict["message_history"] = json.dumps(state_dict["message_history"])

    file_exists = os.path.isfile(tsv_file)
    os.makedirs(
        os.path.dirname(tsv_file) if os.path.dirname(tsv_file) else ".", exist_ok=True
    )

    with open(tsv_file, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        if not file_exists:
            writer.writerow(state_dict.keys())

        writer.writerow(state_dict.values())

    print(f"Conversation state appended to {tsv_file}")
    return tsv_file


def parse_args():
    parser = argparse.ArgumentParser(description="Talk to a customer service agent.")
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=None,
        help="Directory to save conversation files (required for saving)",
    )
    parser.add_argument(
        "--tsv",
        "-t",
        type=str,
        default=None,
        help="TSV file to append final conversation state data",
    )
    return parser.parse_args()


def main_cli():
    args = parse_args()
    final_output = asyncio.run(graph.run())

    if args.directory:
        save_conversation(final_output.message_history, args.directory)

    if args.tsv:
        save_state_as_tsv(final_output, args.tsv)

if __name__ == "__main__":
    main_cli()
