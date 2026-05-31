import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evigraph.utils.latex_sanitizer import (
    escape_latex_for_json,
    unescape_latex_from_json,
    sanitize_claim_for_json,
    desanitize_sentence_for_display,
    sanitize_json_response,
    safe_json_loads,
)


class TestEscapeLatexForJson:

    def test_escapes_single_backslash(self):
        assert escape_latex_for_json(r"\frac") == r"\\frac"

    def test_escapes_multiple_backslashes(self):
        result = escape_latex_for_json(r"\mathcal{L} \sigma")
        assert result == r"\\mathcal{L} \\sigma"

    def test_escapes_complex_equation(self):
        latex = r"\mathcal{L}_{cdm} = -ig_s\frac{\tilde{d}}{2}\bar{t}\sigma_{\mu\nu}\gamma_5"
        result = escape_latex_for_json(latex)
        # Count backslashes - 8 LaTeX commands total
        assert result.count("\\\\") == 8

    def test_preserves_non_latex_content(self):
        text = "This is just plain text with no math."
        assert escape_latex_for_json(text) == text

    def test_handles_empty_string(self):
        assert escape_latex_for_json("") == ""

    def test_handles_none_input(self):
        assert escape_latex_for_json(None) is None

    def test_handles_mixed_content(self):
        text = r"Some text \frac{x}{y} more text \bar{z}"
        result = escape_latex_for_json(text)
        assert result == r"Some text \\frac{x}{y} more text \\bar{z}"


class TestUnescapeLatexFromJson:

    def test_unescapes_double_backslash(self):
        assert unescape_latex_from_json(r"\\frac") == r"\frac"

    def test_unescapes_multiple_backslashes(self):
        result = unescape_latex_from_json(r"\\mathcal{L} \\sigma")
        assert result == r"\mathcal{L} \sigma"

    def test_unescapes_complex_equation(self):
        escaped = r"\\mathcal{L}_{cdm} = -ig_s\\frac{\\tilde{d}}{2}\\bar{t}"
        result = unescape_latex_from_json(escaped)
        assert result == r"\mathcal{L}_{cdm} = -ig_s\frac{\tilde{d}}{2}\bar{t}"

    def test_roundtrip_escape_unescape(self):
        original = r"\frac{x}{y} \sigma_{\mu\nu}"
        escaped = escape_latex_for_json(original)
        unescaped = unescape_latex_from_json(escaped)
        assert unescaped == original

    def test_handles_empty_string(self):
        assert unescape_latex_from_json("") == ""

    def test_handles_none_input(self):
        assert unescape_latex_from_json(None) is None


class TestSanitizeClaimForJson:

    def test_sanitizes_text_field(self):
        claim = {"text": r"\frac{x}{y}", "chunk_id": "ch1"}
        result = sanitize_claim_for_json(claim)
        assert result["text"] == r"\\frac{x}{y}"
        assert result["chunk_id"] == "ch1"

    def test_sanitizes_multiple_string_fields(self):
        claim = {
            "text": r"\alpha",
            "reasoning": r"\beta \gamma",
            "chunk_id": "ch1"
        }
        result = sanitize_claim_for_json(claim)
        assert result["text"] == r"\\alpha"
        assert result["reasoning"] == r"\\beta \\gamma"
        assert result["chunk_id"] == "ch1"

    def test_preserves_non_string_fields(self):
        claim = {
            "text": r"\frac",
            "score": 0.95,
            "count": 42,
            "is_valid": True,
        }
        result = sanitize_claim_for_json(claim)
        assert result["score"] == 0.95
        assert result["count"] == 42
        assert result["is_valid"] is True

    def test_sanitizes_nested_dicts(self):
        claim = {
            "text": r"\alpha",
            "metadata": {
                "equation": r"\beta",
                "score": 0.8
            }
        }
        result = sanitize_claim_for_json(claim)
        assert result["text"] == r"\\alpha"
        assert result["metadata"]["equation"] == r"\\beta"
        assert result["metadata"]["score"] == 0.8

    def test_sanitizes_list_of_strings(self):
        claim = {
            "text": r"\frac",
            "related": [r"\alpha", r"\beta"]
        }
        result = sanitize_claim_for_json(claim)
        assert result["related"] == [r"\\alpha", r"\\beta"]

    def test_handles_non_dict_input(self):
        """Non-dict input should be returned as-is."""
        assert sanitize_claim_for_json("not a dict") == "not a dict"
        assert sanitize_claim_for_json(None) is None


class TestDesanitizeSentenceForDisplay:

    def test_desanitizes_text_field(self):
        sentence = {"text": r"\\frac{x}{y}", "chunk_id": "ch1"}
        result = desanitize_sentence_for_display(sentence)
        assert result["text"] == r"\frac{x}{y}"
        assert result["chunk_id"] == "ch1"

    def test_roundtrip_sanitize_desanitize(self):
        original = {
            "text": r"\mathcal{L} = -ig_s\frac{\tilde{d}}{2}",
            "chunk_id": "ch1"
        }
        sanitized = sanitize_claim_for_json(original)
        desanitized = desanitize_sentence_for_display(sanitized)
        assert desanitized == original

    def test_desanitizes_nested_structures(self):
        sentence = {
            "text": r"\\alpha",
            "metadata": {
                "equation": r"\\beta \\gamma"
            }
        }
        result = desanitize_sentence_for_display(sentence)
        assert result["text"] == r"\alpha"
        assert result["metadata"]["equation"] == r"\beta \gamma"


class TestSanitizeJsonResponse:

    def test_escapes_unescaped_backslashes(self):
        response = r'{"text": "\frac{x}{y}"}'
        result = sanitize_json_response(response)
        # Should have escaped the backslash
        assert r"\\frac" in result

    def test_preserves_already_escaped_backslashes(self):
        response = r'{"text": "\\frac{x}{y}"}'
        result = sanitize_json_response(response)
        # Should not become \\\\
        assert result.count("\\\\") <= 1 or r"\\\\frac" not in result

    def test_strips_markdown_fencing(self):
        response = r'''```json
{"text": "hello"}
```'''
        result = sanitize_json_response(response)
        assert result.strip().startswith("{")
        assert not result.startswith("`")

    def test_handles_empty_string(self):
        assert sanitize_json_response("") == ""

    def test_handles_none_input(self):
        assert sanitize_json_response(None) is None


class TestSafeJsonLoads:

    def test_parses_valid_json(self):
        response = '{"sentences": [{"text": "hello"}]}'
        result = safe_json_loads(response)
        assert result["sentences"][0]["text"] == "hello"

    def test_parses_json_with_unescaped_latex(self):
        response = r'{"text": "\frac{x}{y}"}'
        result = safe_json_loads(response)
        # Result should have proper unescaping
        assert "text" in result

    def test_parses_markdown_fenced_json(self):
        response = r'''```json
{"text": "test"}
```'''
        result = safe_json_loads(response)
        assert result["text"] == "test"

    def test_extracts_json_from_surrounding_text(self):
        response = 'Here is the answer: {"text": "found"} That is all.'
        result = safe_json_loads(response)
        assert result["text"] == "found"

    def test_raises_on_completely_invalid_json(self):
        response = "no json here at all"
        with pytest.raises((json.JSONDecodeError, ValueError)):
            safe_json_loads(response)

    def test_roundtrip_with_latex(self):
        original_text = r"\mathcal{L} = -ig_s\frac{\tilde{d}}{2}"
        
        # Create JSON response with escaped LaTeX
        escaped = escape_latex_for_json(original_text)
        json_response = json.dumps({"text": escaped})
        
        # Parse it back
        parsed = safe_json_loads(json_response)
        unescaped = unescape_latex_from_json(parsed["text"])
        
        # Should recover original
        assert unescaped == original_text


class TestCompleteWorkflow:

    def test_physics_equations_workflow(self):
        # LLM returns response with unescaped LaTeX
        llm_response = r'''
{
  "sentences": [
    {
      "text": "The interaction $\mathcal{L}_{cdm} = -ig_s\frac{\tilde{d}}{2}\bar{t}\sigma_{\mu\nu}\gamma_5 G^{\mu\nu}t$ modifies production.",
      "chunk_id": "ch1"
    }
  ]
}
'''
        
        # Parse with safe loader
        data = safe_json_loads(llm_response)
        assert len(data["sentences"]) == 1
        
        # Content should be preserved
        text = data["sentences"][0]["text"]
        assert r"\mathcal{L}_{cdm}" in text
        assert r"\frac{\tilde{d}}{2}" in text
        assert r"\bar{t}" in text

    def test_mixed_content_workflow(self):
        content = {
            "text": r"Recent results show $\sigma = 1.2 \pm 0.1$ pb using method $\alpha_s^2$.",
            "chunk_id": "ch1",
            "citations": [{"ref": r"\cite{Smith2020}", "text": "Smith et al."}]
        }
        
        # Sanitize for JSON
        sanitized = sanitize_claim_for_json(content)
        # Convert to JSON string
        json_str = json.dumps(sanitized)
        # Parse back
        parsed = json.loads(json_str)
        # Desanitize for display
        restored = desanitize_sentence_for_display(parsed)
        
        # Should recover original
        assert restored == content

    def test_complex_decay_equation_workflow(self):
        equation = (
            r"The decay vertex: $\Gamma^\mu_{Wtb} = -\frac{g}{\sqrt{2}} V_{tb}^\star "
            r"\bar{u}(p_b) [\gamma_\mu (f_1^L P_L + f_1^R P_R) - "
            r"i\sigma^{\mu\nu}(p_t - p_b)_\nu (f_2^L P_L + f_2^R P_R)] u(p_t)$"
        )
        
        # Sanitize and create response (using json.dumps which handles escaping)
        escaped_eq = escape_latex_for_json(equation)
        response = json.dumps({
            "sentences": [{
                "text": escaped_eq,
                "chunk_id": "ch1"
            }]
        })
        
        # Parse safely and desanitize
        data = json.loads(response)
        text = desanitize_sentence_for_display(data["sentences"][0])["text"]
        
        # All LaTeX commands should be present after desanitization
        assert r"\Gamma^\mu_{Wtb}" in text
        assert r"\frac{g}{\sqrt{2}}" in text
        assert r"\bar{u}" in text
        assert r"\sigma^{\mu\nu}" in text
