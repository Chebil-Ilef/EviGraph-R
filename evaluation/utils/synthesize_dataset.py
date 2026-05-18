from __future__ import annotations
import argparse
import json
import logging
import os
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
) -> None:
   
    try:
        from deepeval.synthesizer import Synthesizer
        from deepeval.synthesizer.config import EvolutionConfig, EvolutionMethod
    except ImportError as exc:
        raise ImportError("deepeval is not installed. Run: uv add deepeval") from exc

    evo_method = {
        "reasoning":    EvolutionMethod.REASONING,
        "multi_context": EvolutionMethod.MULTI_CONTEXT,
        "concretizing": EvolutionMethod.CONCRETIZING,
    }.get(evolution.lower(), EvolutionMethod.REASONING)

    synthesizer = Synthesizer(
        model=model,
        async_mode=False,
    )

    # Build context list: list of list[str]
    contexts: list[list[str]] = [g["texts"] for g in groups]

    logger.info("Running Synthesizer on %d context groups (evolution=%s)…", len(contexts), evolution)

    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=True,
        max_goldens_per_context=1,
        evolution_config=EvolutionConfig(
            evolutions={evo_method: 1.0},
            num_evolutions=1,
        ),
    )

    # Merge goldens back with context group metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, "w") as out:
        for i, (golden, group) in enumerate(zip(goldens, groups)):
            record = {
                "golden_id": f"g_{i:04d}",
                "category": group["category"],
                "domain": group["domain"],
                "paper_ids": group["paper_ids"],
                "chunk_ids": group["chunk_ids"],
                "input": golden.input,
                "expected_output": golden.expected_output,
                "context": group["texts"],
                "metadata": group.get("metadata", {}),
            }
            out.write(json.dumps(record) + "\n")
            written += 1

    logger.info("Wrote %d goldens to %s", written, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize benchmark goldens via DeepEval")
    parser.add_argument(
        "--groups_dir",
        default=str(_ROOT / "_data" / "benchmark" / "groups"),
        help="Directory containing cat1.jsonl … cat4.jsonl",
    )
    parser.add_argument(
        "--output",
        default=str(_ROOT / "_data" / "benchmark" / "goldens.jsonl"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_ANSWER_GENERATOR_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
    )
    parser.add_argument(
        "--evolution",
        default="reasoning",
        choices=["reasoning", "multi_context", "concretizing"],
    )
    args = parser.parse_args()

    groups = load_groups(Path(args.groups_dir))
    if not groups:
        logger.error("No context groups found in %s", args.groups_dir)
        sys.exit(1)

    model = build_deepeval_model(args.model)
    synthesize(groups, model, Path(args.output), evolution=args.evolution)


if __name__ == "__main__":
    main()
