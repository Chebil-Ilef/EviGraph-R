from __future__ import annotations
import argparse
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()
else:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault("dotenv", dotenv_stub)

from config.settings import GRAPH_CONFIG
from utils.scicite import SciCiteModel, classify_citation


DEFAULT_EXAMPLES = [
    (
        "background_like",
        (
            "As methods for estimating these underlying graphs have matured, "
            "a number of elaborations to basic Gaussian graphical models have been proposed."
        ),
    ),
    (
        "method_like",
        (
            "We propose a graphical EM technique that estimates both the systemic "
            "and category-specific layers jointly."
        ),
    ),
]


def _read_examples_from_file(path: Path) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    for idx, line in enumerate(path.read_text().splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        examples.append((f"file_example_{idx}", text))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the SciCite classification path used by the demo pipeline. "
            "Loads GRAPH_CONFIG.scicite_model_id, runs raw HF inference, and prints "
            "the mapped EviGraph relation."
        )
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Example text to classify. Can be provided multiple times.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Optional file with one example text per line.",
    )
    args = parser.parse_args()

    examples: list[tuple[str, str]] = []
    if args.file:
        examples.extend(_read_examples_from_file(args.file))
    if args.text:
        examples.extend((f"cli_example_{i}", text) for i, text in enumerate(args.text, 1))
    if not examples:
        examples = DEFAULT_EXAMPLES

    print("SciCite repro")
    print("=" * 72)
    print(f"Configured model id : {GRAPH_CONFIG.scicite_model_id}")
    print("Pipeline task       : text-classification")
    print(f"Examples            : {len(examples)}")
    print()

    model = SciCiteModel.get()

    for name, text in examples:
        raw = model._pipe(text[:512], truncation=True)[0]
        mapped_label, mapped_score = classify_citation(text)

        print("-" * 72)
        print(f"Example       : {name}")
        print(f"Input text    : {text}")
        print(f"Raw label     : {raw.get('label')}")
        print(f"Raw score     : {float(raw.get('score', 0.0)):.6f}")
        print(f"Mapped label  : {mapped_label}")
        print(f"Mapped score  : {mapped_score:.6f}")


if __name__ == "__main__":
    main()
