import pytest

from converser.parser import filter_json_field

# Helper function to create an async token stream from a list of tokens
async def create_token_stream(tokens):
    for token in tokens:
        yield token


# Helper function to collect all yielded values from the filter
async def collect_filtered_values(token_stream, field_name):
    result = []
    async for value in filter_json_field(token_stream, field_name):
        result.append(value)
    return "".join(result)


@pytest.mark.asyncio
async def test_basic_field_extraction():
    tokens = ['{"', 'first', '_name":', '"John",', '"last_name":', '"Doe"}']
    field_name = "first_name"
    
    stream = create_token_stream(tokens)
    result = await collect_filtered_values(stream, field_name)
    
    assert result == 'John'


@pytest.mark.asyncio
async def test_split_field_name():
    tokens = ['{"first_na', 'me":"George",', '"age": 30}']
    field_name = "first_name"
    
    stream = create_token_stream(tokens)
    result = await collect_filtered_values(stream, field_name)
    
    assert result == 'George'


@pytest.mark.asyncio
async def test_complex_value_with_commas():
    tokens = ['{"address":', '"123 Main St, Apt 4",', '"city": "New York"}']
    field_name = "address"
    
    stream = create_token_stream(tokens)
    result = await collect_filtered_values(stream, field_name)
    
    assert result == '123 Main St, Apt 4'


@pytest.mark.asyncio
async def test_escaped_quotes_in_value():
    tokens = ['{"message":', '"He said \\"Hello\\" to me",', '"time": "2023-01-01"}']
    field_name = "message"
    
    stream = create_token_stream(tokens)
    result = await collect_filtered_values(stream, field_name)
    
    assert result == 'He said \\"Hello\\" to me'


@pytest.mark.asyncio
async def test_multiple_occurrences():
    tokens = [
        '{"items": [',
        '{"id": 1, "name": "Item 1"},',
        '{"id": 2, "name": "Item 2"}',
        '],',
        '"name": "Collection"}'
    ]
    field_name = "name"
    
    stream = create_token_stream(tokens)
    result = await collect_filtered_values(stream, field_name)
    
    # This should only find the top-level "name" field, not the nested ones
    assert result == 'Collection'


@pytest.mark.asyncio
async def test_missing_field():
    tokens = ['{"id": 123,', '"title": "Test"}']
    field_name = "description"
    
    stream = create_token_stream(tokens)
    result = await collect_filtered_values(stream, field_name)
    
    assert result == ''


@pytest.mark.asyncio
async def test_very_fragmented_tokens():
    tokens = ['{', '"', 'c', 'o', 'm', 'p', 'l', 'e', 'x', '_', 'f', 'i', 'e', 'l', 'd', '"', ':', '"', 'v', 'a', 'l', 'u', 'e', '"', '}']
    field_name = "complex_field"
    
    stream = create_token_stream(tokens)
    result = await collect_filtered_values(stream, field_name)
    
    assert result == 'value'


