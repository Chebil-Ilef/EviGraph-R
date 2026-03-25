from __future__ import annotations
import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)



_SYSTEM = (
    "You are a scientific information retrieval expert building a retrieval "
    "benchmark for academic papers. Your task is to generate evaluation queries."
)

_USER_TMPL = """\
Given the following chunk from a scientific paper, generate exactly 2 questions.

Requirements for each question:
- Specific to the content of THIS chunk (a researcher must read it to answer)
- Answerable from the chunk without external knowledge
- Tests a key claim, method, result, or definition in the chunk
- NOT overly generic (avoid bare "What is X?" unless X is precisely defined here)

Paper chunk:
---
{text}
---

Respond with ONLY valid JSON, no commentary:
{{"questions": ["<question 1>", "<question 2>"], "answer_strings": ["<key phrase 1>", "<key phrase 2>"]}}
"""


def _build_openai_client():

    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    api_base = (os.getenv("LLM_API_BASE") or "").strip().rstrip("/") or None

    if not api_key:
        return None

    try:
        import openai
    except ImportError:
        logger.warning("openai package not installed; falling back to heuristics.")
        return None

    kwargs: dict = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base

    return openai.OpenAI(**kwargs)


def _resolve_model() -> str:
    model = (os.getenv("LLM_MODEL") or "").strip()
    if not model:
        return "gpt-4o-mini"

    if model.startswith("openai/"):
        model = model[len("openai/"):]
    return model

def _call_llm(
    client,
    model: str,
    chunk: dict,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> tuple[list[str], list[str]]:


    text = chunk.get("embed_text", "")[:1200]  # stay within context window
    prompt = _USER_TMPL.format(text=text)

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=350,
                timeout=60,
            )
            raw = resp.choices[0].message.content or ""

            # Extract JSON even if the model wraps it in markdown fences
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON found in response: {raw[:200]!r}")

            raw_json = match.group()
            # Fix unescaped backslashes (e.g. LaTeX \alpha → \\alpha) before JSON parsing
            raw_json = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw_json)
            data = json.loads(raw_json)
            questions = [q for q in data.get("questions", []) if isinstance(q, str) and q.strip()][:2]
            answers = [a for a in data.get("answer_strings", []) if isinstance(a, str) and a.strip()][:2]
            return questions, answers

        except Exception as exc:
            if attempt < retries - 1:
                wait = retry_delay * (attempt + 1)
                logger.debug("LLM attempt %d failed (%s); retrying in %.1fs", attempt + 1, exc, wait)
                time.sleep(wait)
            else:
                logger.warning(
                    "LLM failed for chunk %s after %d attempts: %s",
                    chunk.get("chunk_uid", "?"), retries, exc,
                )
    return [], []


def _generate_ai_records(
    chunks: list[dict],
    client,
    model: str,
    workers: int,
) -> list[dict]:

    qa_records: list[dict] = []
    question_id_counter = 0
    failed = 0

    logger.info(
        "Generating AI gold Q&A for %d chunks using model=%s workers=%d",
        len(chunks), model, workers,
    )

    def _process(chunk: dict) -> list[dict]:
        questions, answers = _call_llm(client, model, chunk)
        records = []
        for i, question in enumerate(questions):
            if not question:
                continue
            ans_strings = [answers[i]] if i < len(answers) else []
            # Always include the first sentence of the chunk as a fallback answer
            first_sent = chunk.get("embed_text", "").split(".")[0].strip()
            if first_sent and first_sent not in ans_strings:
                ans_strings.append(first_sent[:150])
            records.append({
                "_chunk": chunk,
                "question": question,
                "answer_strings": ans_strings,
            })
        return records

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, c): c for c in chunks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 50 == 0 or done == len(chunks):
                logger.info("  %d / %d chunks processed", done, len(chunks))
            try:
                for item in fut.result():
                    chunk = item["_chunk"]
                    qid = f"q_{question_id_counter:06d}"
                    question_id_counter += 1
                    qa_records.append({
                        "question_id": qid,
                        "query": item["question"],
                        "gold_paper_id": chunk.get("paper_id", ""),
                        "gold_chunk_uid": chunk.get("chunk_uid", chunk.get("uid", "")),
                        "gold_section_title": chunk.get("section_title", ""),
                        "gold_answer_strings": item["answer_strings"],
                    })
            except Exception as exc:
                failed += 1
                logger.warning("Future failed: %s", exc)

    logger.info(
        "AI generation complete: %d Q&A pairs from %d chunks (%d chunks failed)",
        len(qa_records), len(chunks), failed,
    )
    return qa_records

def _heuristic_generate(chunks: list[dict]) -> list[dict]:
  
    logger.warning(
        "No LLM_API_KEY set — falling back to heuristic QA generation. "
        "Set LLM_API_KEY and LLM_API_BASE for AI-quality gold labels."
    )
    qa_records = []
    qid = 0
    for idx, chunk in enumerate(chunks):
        text = chunk.get("embed_text", "")
        if not text or len(text) < 20:
            continue
        sentences = [s.strip() for s in text.split(".")[:2] if s.strip() and len(s.strip()) > 10]
        for phrase in sentences:
            phrase_clean = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", phrase).strip()
            if not phrase_clean or len(phrase_clean) < 15:
                continue
            qa_records.append({
                "question_id": f"q_{qid:06d}",
                "query": f"What is {phrase_clean[:80]}?",
                "gold_paper_id": chunk.get("paper_id", ""),
                "gold_chunk_uid": chunk.get("chunk_uid", chunk.get("uid", f"chunk_{idx}")),
                "gold_section_title": chunk.get("section_title", ""),
                "gold_answer_strings": [phrase_clean[:100]],
            })
            qid += 1
    logger.info("Heuristic fallback: %d Q&A records from %d chunks", len(qa_records), len(chunks))
    return qa_records


def load_chunks(path: Path, sample_size: Optional[int] = None) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
                if sample_size and len(chunks) >= sample_size:
                    break
    logger.info("Loaded %d chunks from %s", len(chunks), path)
    return chunks


def save_qa_records(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved %d Q&A records to %s", len(records), path)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build an AI-generated gold Q&A dataset from unarXiv chunks. "
            "Requires LLM_API_KEY and LLM_API_BASE env vars; falls back to "
            "heuristics if not set."
        )
    )
    p.add_argument("--chunks", required=True, type=Path, help="Input chunks JSONL.")
    p.add_argument("--output", required=True, type=Path, help="Output synthetic_qa JSONL.")
    p.add_argument("--sample-size", type=int, default=None, help="Limit to first N chunks.")
    p.add_argument(
        "--workers", type=int, default=4,
        help="Parallel LLM workers (default 4; 0 = sequential).",
    )
    p.add_argument(
        "--model", type=str, default=None,
        help="Override LLM_MODEL env var (e.g. gpt-4o-mini).",
    )
    args = p.parse_args()

    chunks = load_chunks(args.chunks, sample_size=args.sample_size)
    if not chunks:
        logger.error("No chunks loaded. Check --chunks path.")
        sys.exit(1)

    client = _build_openai_client()
    if client is None:
        records = _heuristic_generate(chunks)
    else:
        model = args.model or _resolve_model()
        workers = max(1, args.workers)
        records = _generate_ai_records(chunks, client, model, workers)

    if not records:
        logger.error("No Q&A records generated. Aborting.")
        sys.exit(1)

    save_qa_records(records, args.output)
    logger.info("Done. %d questions ready for evaluation.", len(records))


if __name__ == "__main__":
    main()
