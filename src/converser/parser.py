from typing import Any, AsyncIterator
from pydantic import TypeAdapter


async def filter_json_field(
    token_stream: AsyncIterator[str], target_field_name: str
) -> AsyncIterator[str]:
    ta = TypeAdapter(dict[str, Any])
    buffer = ""
    last_size = 0
    done = False
    async for token in token_stream:
        if done or not token:
            continue  # Exhaust the iterator
        buffer += token
        if token == " ":
            continue
        parsed = ta.validate_json(buffer, experimental_allow_partial=True)
        response = parsed.get("response") or parsed
        text = response.get(target_field_name)
        if text:
            new_size = len(text)
            if new_size == last_size:
                done = True
            else:
                yield text[last_size:]
                last_size = new_size
