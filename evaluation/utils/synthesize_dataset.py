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


_NFQ_STYLES: list[dict] = [
    {
        "name": "REASON",
        "description": "Causal — asks why something happens or what causes an outcome",
        "patterns": "Why does …? / What causes …? / What is the reason for …? / How come … happened?",
        "example": "Why does sparse retrieval fail on long-tail scientific queries compared to dense retrieval?",
    },
    {
        "name": "EVIDENCE-BASED",
        "description": "Definitional/descriptive — asks what something is or how it works",
        "patterns": "What is …? / How does … work? / What are the properties of …? / How do you describe …?",
        "example": "How does cross-encoder reranking improve retrieval precision in hybrid search pipelines?",
    },
    {
        "name": "COMPARISON",
        "description": "Comparative — asks about differences or similarities between two things",
        "patterns": "How is X … compared to Y? / What are the differences between X and Y? / How does X perform against Y?",
        "example": "How does the evidence graph approach differ from flat-chunk retrieval in handling cross-paper evidence?",
    },
    {
        "name": "INSTRUCTION",
        "description": "Methodological — asks how to do or achieve something",
        "patterns": "How to …? / How can … be achieved? / What is the process for …? / What is the best way to …?",
        "example": "How can citation expansion be used to improve claim attribution in multi-paper synthesis?",
    },
    {
        "name": "EXPERIENCE",
        "description": "Evaluative — asks about advantages, limitations, or practical assessment of a method or system",
        "patterns": "What are the advantages of …? / What are the limitations of …? / How well does … perform in practice? / Should … be used for …?",
        "example": "What are the limitations of BM25-based sparse retrieval for scientific literature search?",
    },
    {
        "name": "FACTOID",
        "description": "Factual — asks for a specific, concise fact, value, or entity name",
        "patterns": "What is the value of …? / Who proposed …? / Which method achieves …? / What does … stand for?",
        "example": "What accuracy does the proposed model achieve on the benchmark described in the results section?",
    },
    {
        "name": "DEBATE",
        "description": "Argumentative — asks whether a claim holds, a method truly works, or an approach is valid, supported by pros and cons from the evidence",
        "patterns": "Does … exist? / Can … be successful? / Is … really …? / Does … actually improve …?",
        "example": "Does citation-aware evidence expansion actually improve answer attribution compared to flat retrieval?",
    },
]

# Cycle through styles deterministically so each batch of 4 groups covers all styles.
def _nfq_style_for_index(i: int) -> dict:
    return _NFQ_STYLES[i % len(_NFQ_STYLES)]


_CAT3_BRIDGE_CONSTRAINT = (
    "IMPORTANT: The two passages come from two different papers connected by a citation. "
    "Passage A cites Passage B's paper. Generate a question that ties a specific claim, "
    "method, or result stated in Passage A to the work introduced or established in Passage B. "
    "The question MUST be unanswerable from Passage A alone (A only asserts or uses the result) "
    "and unanswerable from Passage B alone (B lacks the citing context from A). "
    "The citation link is the bridge — the question must require both ends to answer."
)

_CAT4_MULTISOURCE_CONSTRAINT = (
    "IMPORTANT: The three passages come from three different papers on the same theme. "
    "Generate a question that requires connecting or contrasting information across all three passages. "
    "The question MUST NOT be fully answerable from any single passage alone."
)

_CAT_CONSTRAINTS: dict[int, str] = {
    3: _CAT3_BRIDGE_CONSTRAINT,
    4: _CAT4_MULTISOURCE_CONSTRAINT,
}


def _nfq_input_format(style: dict, category: int | None = None) -> str:
    base = (
        f"A specific, self-contained research question in English following the "
        f"'{style['name']}' question style ({style['description']}). "
        f"Typical patterns: {style['patterns']}. "
        f"Example: \"{style['example']}\". "
        f"The question must be understandable without reading the source text, "
        f"must NOT use the phrase 'relationship between X and Y', "
        f"and must require synthesis across the provided passages to answer fully."
    )
    constraint = _CAT_CONSTRAINTS.get(category or 0, "")
    return f"{base} {constraint}".strip() if constraint else base


def _is_usable_golden(input_text: str | None, expected_output: str | None) -> tuple[bool, str]:

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


_SINGLE_HOP_PROMPT = (
    "Given ONLY the following passage, can you fully and correctly answer the question below?\n"
    "Answer with exactly YES or NO on the first line, then one sentence of justification.\n\n"
    "Passage:\n{chunk}\n\nQuestion:\n{question}"
)


def _is_single_hop_shortcut(question: str, chunks: list[str], model) -> bool:
    """Return True if any single chunk is sufficient to fully answer the question.

    Implements the MuSiQue single-hop shortcut filter (Trivedi et al., TACL 2022).
    Conservative: a YES from any chunk triggers discard.
    """
    for chunk in chunks:
        prompt = _SINGLE_HOP_PROMPT.format(chunk=chunk[:2000], question=question)
        try:
            reply: str = model.generate(prompt).strip()
        except Exception as exc:
            logger.debug("Single-hop filter LLM call failed: %s", exc)
            continue
        if reply.upper().startswith("YES"):
            return True
    return False


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

    Question styles are rotated across groups following the NFQ taxonomy
    (Kiesel et al., SIGIR 2022) to ensure diversity beyond "relationship between X and Y".

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
    primary = evo_map.get(evolution.lower(), Evolution.REASONING)
    if primary == Evolution.REASONING:
        evolutions = {Evolution.REASONING: 0.6, Evolution.CONCRETIZING: 0.4}
    else:
        evolutions = {primary: 1.0}

    # Group contexts by category so style cycling happens independently per category.
    # This guarantees exactly `per_style` goldens per (category, style) pair.
    by_cat: dict[int, list[int]] = {}  # cat → list of original group indices
    for i, g in enumerate(groups):
        by_cat.setdefault(g["category"], []).append(i)

    n_styles = len(_NFQ_STYLES)

    # per_style[cat] = how many goldens to keep per style in that category
    # Derived from targets: target must be divisible by n_styles.
    per_style: dict[int, int] = {}
    if targets:
        for cat, t in targets.items():
            if t % n_styles != 0:
                raise ValueError(
                    f"Category {cat} target {t} is not divisible by {n_styles} (number of NFQ styles). "
                    f"Adjust target to a multiple of {n_styles}."
                )
            per_style[cat] = t // n_styles

    logger.info(
        "Running Synthesizer on %d context groups (evolution=%s), "
        "%d NFQ styles, per-style quota: %s",
        len(groups), evolution, n_styles,
        {cat: per_style.get(cat, "unlimited") for cat in sorted(by_cat)},
    )

    # all_goldens[original_group_index] = (golden, nfq_style_name)
    all_goldens: list = [None] * len(groups)

    # Checkpoint: resume from a .partial file if one exists from a previous run.
    # Each line: {"orig_idx": int, "style": str, "input": str, "expected_output": str}
    checkpoint_path = output_path.with_suffix(output_path.suffix + ".partial")
    done_slices: set[tuple[int, int]] = set()  # (cat, style_idx) pairs already completed

    if checkpoint_path.exists():
        logger.info("Resuming from checkpoint %s", checkpoint_path)
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                orig_idx = rec["orig_idx"]

                class _Stub:
                    pass
                stub = _Stub()
                stub.input = rec["input"]
                stub.expected_output = rec["expected_output"]
                all_goldens[orig_idx] = (stub, rec["style"])
                done_slices.add((rec["cat"], rec["style_idx"]))
        logger.info("Loaded %d goldens from checkpoint (%d slices done)",
                    sum(1 for g in all_goldens if g is not None), len(done_slices))

    # FiltrationConfig is disabled: it requires the model to return structured
    # JSON for InputFeedback/RewrittenInput schemas, which our custom OpenAI
    # wrapper cannot guarantee reliably. Quality filtering is done post-hoc
    # via _is_usable_golden instead.
    for cat, cat_indices in sorted(by_cat.items()):
        # Assign styles by cycling within this category's groups only.
        # cat_indices[0] → style 0, cat_indices[1] → style 1, …, cat_indices[7] → style 0, …
        style_to_indices: dict[int, list[int]] = {}
        for pos, orig_idx in enumerate(cat_indices):
            s = pos % n_styles
            style_to_indices.setdefault(s, []).append(orig_idx)

        for style_idx, style in enumerate(_NFQ_STYLES):
            if (cat, style_idx) in done_slices:
                logger.info("[NFQ] cat=%d style=%s — skipping (already in checkpoint)", cat, style["name"])
                continue

            slice_orig_indices = style_to_indices.get(style_idx, [])
            if not slice_orig_indices:
                continue

            slice_contexts = [groups[i]["texts"] for i in slice_orig_indices]

            styling_config = StylingConfig(
                input_format=_nfq_input_format(style, category=cat),
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

            logger.info(
                "[NFQ] cat=%d style=%s (%d/%d) — generating %d goldens",
                cat, style["name"], style_idx + 1, n_styles, len(slice_contexts),
            )
            slice_goldens = synthesizer.generate_goldens_from_contexts(
                contexts=slice_contexts,
                include_expected_output=True,
                max_goldens_per_context=1,
            )

            # Save to checkpoint immediately after each slice succeeds
            with open(checkpoint_path, "a") as ckpt:
                for orig_idx, golden in zip(slice_orig_indices, slice_goldens):
                    all_goldens[orig_idx] = (golden, style["name"])
                    ckpt.write(json.dumps({
                        "orig_idx": orig_idx,
                        "cat": cat,
                        "style_idx": style_idx,
                        "style": style["name"],
                        "input": golden.input,
                        "expected_output": golden.expected_output,
                    }) + "\n")

    # Bucket accepted goldens per category, enforcing per-style quota.
    # style_counts[cat][style_name] tracks how many of each style we've kept.
    buckets: dict[int, list[dict]] = {}
    style_counts: dict[int, dict[str, int]] = {}
    skipped = 0

    for i, (group, golden_tuple) in enumerate(zip(groups, all_goldens)):
        if golden_tuple is None:
            skipped += 1
            continue
        golden, nfq_style = golden_tuple
        cat = group["category"]

        usable, reason = _is_usable_golden(golden.input, golden.expected_output)
        if not usable:
            logger.warning(
                "Dropping golden %d (cat=%d domain=%s style=%s) — %s: %r",
                i, cat, group["domain"], nfq_style, reason, (golden.input or "")[:80],
            )
            skipped += 1
            continue

        if cat in (3, 4) and _is_single_hop_shortcut(golden.input, group["texts"], model):
            logger.warning(
                "Dropping golden %d (cat=%d domain=%s style=%s) — single-hop shortcut: %r",
                i, cat, group["domain"], nfq_style, golden.input[:80],
            )
            skipped += 1
            continue

        # Enforce per-style quota within this category
        quota = per_style.get(cat)
        if quota is not None:
            counts = style_counts.setdefault(cat, {})
            if counts.get(nfq_style, 0) >= quota:
                continue  # already have enough of this style for this category
            counts[nfq_style] = counts.get(nfq_style, 0) + 1

        record = {
            "category": cat,
            "domain": group["domain"],
            "nfq_style": nfq_style,
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
            counts = style_counts.get(cat, {})
            quota = per_style.get(cat)
            # Warn if any style is short of its quota
            if quota is not None:
                for style in _NFQ_STYLES:
                    n = counts.get(style["name"], 0)
                    if n < quota:
                        logger.warning(
                            "Category %d style=%s: only %d/%d goldens — "
                            "increase OVERSAMPLE_FACTOR or check context group quality",
                            cat, style["name"], n, quota,
                        )
            if target and got < target:
                logger.warning(
                    "Category %d: only %d/%d good goldens after filtering — "
                    "consider increasing the oversampling buffer (OVERSAMPLE_FACTOR)",
                    cat, got, target,
                )
            style_summary = {s["name"]: counts.get(s["name"], 0) for s in _NFQ_STYLES}
            logger.info("Category %d: %d goldens — style breakdown: %s", cat, got, style_summary)
            for record in buckets[cat]:
                record["golden_id"] = f"g_{written:04d}"
                out.write(json.dumps(record) + "\n")
                written += 1

    if skipped:
        logger.warning("Dropped %d degenerate goldens total", skipped)
    logger.info("Wrote %d goldens to %s", written, output_path)

    # Remove checkpoint now that the final output is written
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed checkpoint %s", checkpoint_path)


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
