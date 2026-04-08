import json
import re
from typing import Any


def escape_latex_for_json(text: str) -> str:
 
    if not text or not isinstance(text, str):
        return text
    
    # backslashes: \ → \\
    return text.replace("\\", "\\\\")


def unescape_latex_from_json(text: str) -> str:

    if not text or not isinstance(text, str):
        return text
    
    # Unescape: \\ → \
    return text.replace("\\\\", "\\")


def sanitize_claim_for_json(claim: dict[str, Any]) -> dict[str, Any]:

    if not isinstance(claim, dict):
        return claim
    
    sanitized = {}
    for key, value in claim.items():
        if isinstance(value, str):
            sanitized[key] = escape_latex_for_json(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_claim_for_json(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_claim_for_json(item) if isinstance(item, dict)
                else escape_latex_for_json(item) if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            sanitized[key] = value
    
    return sanitized


def desanitize_sentence_for_display(sentence: dict[str, Any]) -> dict[str, Any]:

    if not isinstance(sentence, dict):
        return sentence
    
    desanitized = {}
    for key, value in sentence.items():
        if isinstance(value, str):
            desanitized[key] = unescape_latex_from_json(value)
        elif isinstance(value, dict):
            desanitized[key] = desanitize_sentence_for_display(value)
        elif isinstance(value, list):
            desanitized[key] = [
                desanitize_sentence_for_display(item) if isinstance(item, dict)
                else unescape_latex_from_json(item) if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            desanitized[key] = value
    
    return desanitized


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
