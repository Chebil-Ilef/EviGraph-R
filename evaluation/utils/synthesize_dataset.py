from __future__ import annotations
import argparse
import json
import logging
import os
import re as _re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ENV_FILE = _ROOT / ".env"
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from evaluation.utils import build_deepeval_model

_FORMULA_RE = _re.compile(r"\{\{formula:[^}]+\}\}")
_REF_RE = _re.compile(r"\bREF\b")
_CODE_FENCE_RE = _re.compile(r"```")
_MIN_INPUT_LEN = 60    # characters — below this the question is too vague/narrow
_MAX_INPUT_LEN = 400   # characters — above this it's likely an LLM monologue, not a question
_MIN_OUTPUT_LEN = 80   # characters


def _is_usable_golden(input_text: str | None, expected_output: str | None) -> tuple[bool, str]:
    """Return (usable, reason). reason is empty when usable."""
    q = (input_text or "").strip()
    a = (expected_output or "").strip()
    if len(q) < _MIN_INPUT_LEN:
        return False, f"question too short ({len(q)} < {_MIN_INPUT_LEN})"
    if len(q) > _MAX_INPUT_LEN:
        return False, f"question too long ({len(q)} > {_MAX_INPUT_LEN}) — likely LLM monologue"
    if len(a) < _MIN_OUTPUT_LEN:
        return False, f"answer too short ({len(a)} < {_MIN_OUTPUT_LEN})"
    if _CODE_FENCE_RE.search(q) or _CODE_FENCE_RE.search(a):
        return False, "code fence in question or answer — likely malformed LLM output"
    if _FORMULA_RE.search(q) or _FORMULA_RE.search(a):
        return False, "formula placeholder in question or answer"
    if _REF_RE.search(a):
        return False, "unresolved REF tag in answer"
    return True, ""


def load_groups(groups_dir: Path) -> list[dict]:
    groups: list[dict] = []
    for fname in sorted(groups_dir.glob("cat*.jsonl")):
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if line:
                    groups.append(json.loads(line))
        logger.info("Loaded %s → %d groups (running total: %d)", fname.name, sum(1 for _ in open(fname)), len(groups))
    return groups


def synthesize(
    groups: list[dict],
    model,
    output_path: Path,
    evolution: str = "reasoning",
    targets: dict[int, int] | None = None,
) -> None:
    """Generate goldens from context groups.

    targets: optional dict mapping category int → max goldens to keep.
    If provided, output is trimmed per category to exactly that count (or
    as many good goldens as available, with a warning if short).
    """
    try:
        from deepeval.synthesizer import Synthesizer
        from deepeval.synthesizer.config import EvolutionConfig, Evolution, StylingConfig
    except ImportError as exc:
        raise ImportError("deepeval is not installed. Run: uv add deepeval") from exc

    evo_map = {
        "reasoning":     Evolution.REASONING,
        "multi_context": Evolution.MULTICONTEXT,
        "concretizing":  Evolution.CONCRETIZING,
        "constrained":   Evolution.CONSTRAINED,
        "comparative":   Evolution.COMPARATIVE,
    }
    # Mix CONCRETIZING in alongside the chosen evolution to force specificity
    primary = evo_map.get(evolution.lower(), Evolution.REASONING)
    if primary == Evolution.REASONING:
        evolutions = {Evolution.REASONING: 0.6, Evolution.CONCRETIZING: 0.4}
    else:
        evolutions = {primary: 1.0}

    # FiltrationConfig is disabled: it requires the model to return structured
    # JSON for InputFeedback/RewrittenInput schemas, which our custom OpenAI
    # wrapper cannot guarantee reliably. Quality filtering is done post-hoc
    # via _is_usable_golden instead.
    styling_config = StylingConfig(
        input_format=(
            "A specific, self-contained research question in English that asks about "
            "a concrete concept, method, result, or relationship described in the context. "
            "The question must be understandable without reading the source text and must "
            "require synthesis across the provided passages to answer fully."
        ),
        expected_output_format=(
            "A detailed factual answer in complete sentences that directly addresses the "
            "question using evidence from the context. Do not include formula placeholders, "
            "unresolved reference tags (REF), or LaTeX that cannot be rendered as plain text."
        ),
        task="Scientific literature question answering requiring multi-passage synthesis",
        scenario="A researcher querying a scientific paper retrieval system for evidence-backed answers",
    )

    synthesizer = Synthesizer(
        model=model,
        async_mode=False,
        styling_config=styling_config,
        evolution_config=EvolutionConfig(
            evolutions=evolutions,
            num_evolutions=1,
        ),
    )

    contexts: list[list[str]] = [g["texts"] for g in groups]
    logger.info("Running Synthesizer on %d context groups (evolution=%s)…", len(contexts), evolution)

    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=True,
        max_goldens_per_context=1,
    )

    # Bucket accepted goldens per category, respecting per-category targets
    # category → list of serialisable records (in order)
    buckets: dict[int, list[dict]] = {}
    skipped = 0
    for i, (golden, group) in enumerate(zip(goldens, groups)):
        cat = group["category"]
        usable, reason = _is_usable_golden(golden.input, golden.expected_output)
        if not usable:
            logger.warning(
                "Dropping golden %d (cat=%s domain=%s) — %s: %r",
                i, cat, group["domain"], reason, (golden.input or "")[:80],
            )
            skipped += 1
            continue
        # Respect per-category target cap
        if targets and cat in targets and len(buckets.get(cat, [])) >= targets[cat]:
            continue
        record = {
            "category": cat,
            "domain": group["domain"],
            "paper_ids": group["paper_ids"],
            "chunk_ids": group["chunk_ids"],
            "input": golden.input,
            "expected_output": golden.expected_output,
            "context": group["texts"],
            "metadata": group.get("metadata", {}),
        }
        buckets.setdefault(cat, []).append(record)

    # Write in category order, assign sequential IDs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, "w") as out:
        for cat in sorted(buckets):
            target = targets.get(cat) if targets else None
            got = len(buckets[cat])
            if target and got < target:
                logger.warning(
                    "Category %d: only %d/%d good goldens after filtering — "
                    "consider increasing the oversampling buffer (OVERSAMPLE_FACTOR)",
                    cat, got, target,
                )
            for record in buckets[cat]:
                record["golden_id"] = f"g_{written:04d}"
                out.write(json.dumps(record) + "\n")
                written += 1

    if skipped:
        logger.warning("Dropped %d degenerate goldens total", skipped)
    logger.info("Wrote %d goldens to %s", written, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize benchmark goldens via DeepEval")
    parser.add_argument(
        "--groups_dir",
        default=str(_ROOT / "_data" / "benchmark" / "groups"),
        help="Directory containing cat1.jsonl … cat4.jsonl",
    )
    parser.add_argument("--output", default=str(_ROOT / "_data" / "benchmark" / "goldens.jsonl"))
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_ANSWER_GENERATOR_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
    )
    parser.add_argument(
        "--evolution",
        default="reasoning",
        choices=["reasoning", "multi_context", "concretizing", "constrained", "comparative"],
    )
    parser.add_argument(
        "--cat1_target", type=int, default=None,
        help="Max goldens to keep for category 1 (default: keep all good ones)",
    )
    parser.add_argument("--cat2_target", type=int, default=None)
    parser.add_argument("--cat3_target", type=int, default=None)
    parser.add_argument("--cat4_target", type=int, default=None)
    args = parser.parse_args()

    targets: dict[int, int] | None = None
    raw = {1: args.cat1_target, 2: args.cat2_target, 3: args.cat3_target, 4: args.cat4_target}
    if any(v is not None for v in raw.values()):
        targets = {k: v for k, v in raw.items() if v is not None}

    groups = load_groups(Path(args.groups_dir))
    if not groups:
        logger.error("No context groups found in %s", args.groups_dir)
        sys.exit(1)

    model = build_deepeval_model(args.model)
    synthesize(groups, model, Path(args.output), evolution=args.evolution, targets=targets)


if __name__ == "__main__":
    main()
