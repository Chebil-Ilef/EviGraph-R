import json
import re
from typing import Any, TypeAlias


JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | dict[str, "JsonValue"] | list["JsonValue"]


def escape_latex_for_json(text: str | None) -> str | None:
 
    if not text or not isinstance(text, str):
        return text
    
    # backslashes: \ → \\
    return text.replace("\\", "\\\\")


def unescape_latex_from_json(text: str | None) -> str | None:

    if not text or not isinstance(text, str):
        return text
    
    # Unescape: \\ → \
    return text.replace("\\\\", "\\")


def _sanitize_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        return escape_latex_for_json(value)

    if isinstance(value, dict):
        return {key: _sanitize_json_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]

    return value


def sanitize_claim_for_json(claim: Any) -> Any:

    if not isinstance(claim, dict):
        return claim

    return {key: _sanitize_json_value(value) for key, value in claim.items()}


def _desanitize_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        return unescape_latex_from_json(value)

    if isinstance(value, dict):
        return {key: _desanitize_json_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_desanitize_json_value(item) for item in value]

    return value


def desanitize_sentence_for_display(sentence: Any) -> Any:

    if not isinstance(sentence, dict):
        return sentence

    return {key: _desanitize_json_value(value) for key, value in sentence.items()}


def sanitize_json_response(llm_response: str) -> str:

    if not llm_response or not isinstance(llm_response, str):
        return llm_response
    
    # Remove markdown fencing if present
    response = llm_response.strip()
    if response.startswith("```"):
        response = response.strip("`").strip()
        if response.startswith("json"):
            response = response[4:].strip()
    
    # Pattern: find unescaped backslashes (not already \\)
    response = re.sub(r'(?<!\\)\\(?!\\)', r'\\\\', response)
    
    return response


def safe_json_loads(text: str) -> Any:

    sanitized = sanitize_json_response(text)
    
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError as e:
        # Try extracting JSON from partial response
        start = sanitized.find("{")
        end = sanitized.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise
        
        extracted = sanitized[start:end + 1]
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            # Last resort: raise original error with context
            raise json.JSONDecodeError(
                f"Failed to parse JSON (tried extraction): {str(e)}",
                text,
                e.pos
            )
