from typing import AsyncIterator
from partialjson.json_parser import JSONParser

async def filter_json_field(token_stream: AsyncIterator[str], target_field_name: str) -> AsyncIterator[str]:
    parser = JSONParser(strict=False)

    buffer = ""
    last_size = 0
    done = False
    async for token in token_stream:
        print(f'token={token}, done={done}', end=" ")
        if done:
            continue # Exhaust the iterator
        buffer += token
        print(f'buffer: {buffer} ', end="")
        if token == " ":
            continue
        parsed = parser.parse(buffer)
        response = parsed.get("response") or parsed
        text = response.get(target_field_name) 
        print(f'text: {text}')
        if text:
            new_size = len(text)
            if new_size == last_size:
                done = True
            else:
                yield text[last_size:]
                last_size = new_size
