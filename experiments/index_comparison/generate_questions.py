from __future__ import annotations
import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"
if _ENV_FILE.exists():
    import os
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# Prompts

_SYSTEM = (
    "You are a scientific information retrieval expert building a retrieval "
    "benchmark for academic papers. Your task is to generate evaluation queries."
)

_ABSTRACT_TMPL = """\
Given the following abstract from a scientific paper, generate exactly 1 question.

Requirements:
- Specific to THIS paper's concrete contribution, result, or method.
- Answerable from this abstract alone.
- Avoid generic questions — ask about a specific claim or result unique to this paper.
- The question must be answerable ONLY by reading this paper (not any similar paper).

Title: {title}
Abstract:
---
{abstract}
---

Respond with ONLY valid JSON, no commentary:
{{"question": "<specific question>", "answer_string": "<key phrase from abstract that answers it>"}}
"""

_SECTION_TMPL = """\
Given the following excerpt from a specific section of a scientific paper, generate exactly 1 \
retrieval evaluation question.

Requirements:
- The question must ONLY be answerable by reading THIS specific section of THIS paper.
- Ask about a concrete claim, number, method name, dataset, architecture detail, or \
experimental finding stated in the section text below.
- Do NOT ask something answerable from the abstract or title alone.
- Phrased as a researcher's natural information need (not "according to this paper…").
- The answer must be a specific phrase or value findable in the section text.

Title: {title}
Section: {section_title}
Section text:
---
{section_text}
---

Respond with ONLY valid JSON, no commentary:
{{"question": "<specific question about content in this section>", "answer_string": "<key phrase from the section text that answers it>"}}
"""

_FULLPAPER_TMPL = """\
Given the title and a methods/results excerpt from a scientific paper, generate a question \
that probes a specific experimental or methodological detail.

Requirements:
- Ask about a specific dataset name, evaluation metric value, architectural choice, \
baseline comparison, or implementation detail.
- NOT answerable from a typical abstract — requires reading the paper body.
- Phrased as a researcher's natural information need.
- The answer must be a specific phrase or value findable in the section text.

Title: {title}
Section: {section_title}
Section text:
---
{section_text}
---

Respond with ONLY valid JSON, no commentary:
{{"question": "<specific methods/results question>", "answer_string": "<expected answer key phrase>"}}
"""


def _build_llm_client():
    try:
        from utils.llm import LLMClient
        from config.settings import LLM
        if not LLM.api_key:
            return None, ""
        client = LLMClient()
        model = LLM.answer_generator_model  # Llama-3.3-70B-Instruct from .env
        logger.info("Using LLMClient with model: %s", model)
        return client, model
    except Exception as exc:
        logger.warning("LLMClient init failed (%s) — using heuristic fallback.", exc)
        return None, ""


def _call_llm(
    client,
    model: str,
    prompt: str,
    label: str,
    max_tokens: int = 300,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    for attempt in range(retries):
        try:
            result = client.chat_text(
                model=model,
                system_prompt=_SYSTEM,
                user_prompt=prompt,
                temperature=0.3,
                max_tokens=max_tokens,
                timeout=120,
            )
            time.sleep(1.0)  # 1s between calls to avoid rate limiting
            return result
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                logger.warning("LLM failed for %s after %d attempts: %s", label, retries, exc)
    return ""


def _parse_json(raw: str) -> object:
    # Try strict parse first (array, then object)
    for pattern in [r"\[.*\]", r"\{.*\}"]:
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', match.group())
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

    # Truncated array recovery: extract all complete {...} objects from a partial list
    objects = []
    for m in re.finditer(r'\{[^{}]*\}', raw, re.DOTALL):
        cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', m.group())
        try:
            objects.append(json.loads(cleaned))
        except json.JSONDecodeError:
            pass
    if objects:
        return objects

    return None


def _fetch_section_texts(paper_ids: list[str]) -> dict[str, tuple[str, str]]:
    """Return {paper_id: (section_title, section_text)} for the best body section of each paper."""
    try:
        from qdrant_client import QdrantClient
        from config.settings import QDRANT_ACTIVE, QDRANT_CONNECTION

        conn = QDRANT_CONNECTION
        if conn.url:
            client = QdrantClient(url=conn.url, api_key=conn.api_key, timeout=conn.timeout)
        else:
            client = QdrantClient(host=conn.host, port=conn.port, timeout=conn.timeout)
        collection = QDRANT_ACTIVE.collection_name

        result: dict[str, tuple[str, str]] = {}
        # Body sections only — strictly no abstract so questions can't be abstract-answerable
        preferred_sections = ["methods", "results", "methodology", "experiments",
                              "discussion", "evaluation", "approach", "model", "system"]

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        for paper_id in paper_ids:
            hits, _ = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="paper_id_arxiv", match=MatchValue(value=paper_id)),
                        FieldCondition(key="chunk_type", match=MatchValue(value="subsection")),
                    ]
                ),
                limit=50,
                with_payload=True,
                with_vectors=False,
            )
            if not hits:
                continue

            best_text = ""
            best_title = ""
            best_priority = len(preferred_sections) + 1

            for hit in hits:
                payload = hit.payload or {}
                section = (payload.get("imrad_section_title") or payload.get("section_title") or "").lower()
                text = payload.get("embed_text", "")
                if not text or len(text) < 150:
                    continue
                priority = len(preferred_sections)
                for i, pref in enumerate(preferred_sections):
                    if pref in section:
                        priority = i
                        break
                if priority < best_priority:
                    best_priority = priority
                    best_text = text
                    best_title = payload.get("imrad_section_title") or payload.get("section_title") or section

            if best_text:
                # Strip unresolved citation/figure/table/equation placeholders that
                # confuse the LLM into treating "REF" as a shared technical concept.
                clean = re.sub(r'\b(REF|FORMULA|EQUATION|FIG|TABLE)\b', '', best_text)
                clean = re.sub(r'\s{2,}', ' ', clean).strip()
                result[paper_id] = (best_title, clean[:1200])

        client.close()
        return result

    except Exception as exc:
        logger.error("Qdrant section fetch failed: %s", exc, exc_info=True)
        return {}


def _heuristic_single(title: str, section_title: str, section_text: str, source: str) -> tuple[str, str]:
    sentences = [s.strip() for s in section_text.split(".") if len(s.strip()) > 30]
    if source == "section":
        text = sentences[1] if len(sentences) > 1 else sentences[0] if sentences else section_text[:200]
        return f"What specific method or result is described in the {section_title} section of '{title[:60]}'?", text[:120]
    text = sentences[-1] if len(sentences) > 1 else section_text[-200:]
    return f"What experimental detail is reported in the {section_title} section of '{title[:60]}'?", text[:120]


def _generate_single(paper: dict, client, model: str,
                     section_texts: dict[str, tuple[str, str]]) -> list[dict]:
    pid = paper["paper_id"]
    title = paper["title"]

    section_info = section_texts.get(pid)
    if not section_info:
        # No body section available — skip this paper entirely rather than fall back to abstract
        logger.warning("No body section text for %s ('%s') — skipping single-paper questions.", pid, title[:60])
        return []

    section_title, section_text = section_info
    records = []

    # Three question types: 1 section-specific, 1 fullpaper, 1 abstract (kept small for fairness).
    # Section and fullpaper are the primary eval signal; abstract is kept to show EviGraph-R
    # still wins even on SQuAI's home turf.
    abstract = paper.get("abstract", "")[:1500]

    for source, tmpl in [("section", _SECTION_TMPL), ("fullpaper", _FULLPAPER_TMPL), ("abstract", _ABSTRACT_TMPL)]:
        if client:
            if source == "abstract":
                prompt = tmpl.format(title=title, abstract=abstract)
            else:
                prompt = tmpl.format(title=title, section_title=section_title, section_text=section_text)
            raw = _call_llm(client, model, prompt, f"{pid}/{source}", max_tokens=250)
            data = _parse_json(raw)
            question = str(data.get("question", "")).strip() if isinstance(data, dict) else ""
            answer = str(data.get("answer_string", "")).strip() if isinstance(data, dict) else ""
        else:
            question, answer = _heuristic_single(title, section_title, section_text, source)

        if question:
            records.append({
                "type": "single",
                "source": source,
                "gold_paper_ids": [pid],
                "gold_titles": [title],
                # gold_section only set for section/fullpaper — abstract questions have no section target
                "gold_section": section_title if source != "abstract" else None,
                "domain": paper["domain"],
                "cluster": paper["cluster"],
                "question": question,
                "answer_string": answer,
            })
    return records


def generate_questions(
    papers_path: str,
    output_path: str,
    workers: int = 4,
) -> None:
    papers = []
    with open(papers_path) as f:
        for line in f:
            papers.append(json.loads(line))
    logger.info("Loaded %d papers from %s", len(papers), papers_path)

    client, model = _build_llm_client()
    if not client:
        logger.warning("No LLM client — using heuristic fallback.")

    # Fetch rich section texts from Qdrant for all papers (best-effort)
    all_paper_ids = [p["paper_id"] for p in papers]
    logger.info("Fetching section texts from Qdrant for %d papers …", len(all_paper_ids))
    section_texts = _fetch_section_texts(all_paper_ids)
    logger.info("  Got section texts for %d / %d papers", len(section_texts), len(all_paper_ids))

    single_raw: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_generate_single, p, client, model, section_texts): p for p in papers}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 10 == 0 or done == len(papers):
                logger.info("  [single] %d / %d papers processed", done, len(papers))
            try:
                single_raw.extend(fut.result())
            except Exception as exc:
                logger.warning("Single-paper future failed: %s", exc)

    # Write output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as out:
        for i, item in enumerate(single_raw):
            record = {
                "question_id": f"q_{i:04d}",
                "query": item["question"],
                "type": item["type"],
                "source": item["source"],
                "gold_paper_ids": item["gold_paper_ids"],
                "gold_titles": item["gold_titles"],
                "gold_section": item.get("gold_section"),  # section title for Section Hit@k
                "domain": item["domain"],
                "cluster": item["cluster"],
                "answer_string": item["answer_string"],
            }
            out.write(json.dumps(record) + "\n")

    logger.info("Wrote %d questions to %s", len(single_raw), output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate single-paper evaluation questions"
    )
    parser.add_argument(
        "--papers",
        default="experiments/index_comparison/results/sampled_papers.jsonl",
    )
    parser.add_argument(
        "--output",
        default="experiments/index_comparison/results/questions.jsonl",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    generate_questions(
        papers_path=args.papers,
        output_path=args.output,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
