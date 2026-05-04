from __future__ import annotations

import argparse
import json
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from utils.nli import NLIModel, nli_verify


DEFAULT_EXAMPLES = [
    {
        "id": "direct_support",
        "claim": "The Bayesian reinforcement learning framework is extended to inverse RL.",
        "evidence": (
            "In this section, we discuss extensions of the Bayesian reinforcement learning "
            "(BRL) framework to the following classes of problems: PAC-Bayes model selection, "
            "inverse RL, multi-agent RL, and multi-task RL."
        ),
    },
    {
        "id": "direct_support_exact_copy",
        "claim": "They also provide an elegant approach to action selection in the face of uncertainty.",
        "evidence": (
            "Bayesian methods for machine learning have been widely investigated, yielding principled "
            "methods for incorporating prior information into inference algorithms. They also provide "
            "an elegant approach to action selection in the face of uncertainty."
        ),
    },
    {
        "id": "paraphrase_support",
        "claim": "Bayesian methods provide an elegant approach to action-selection as a function of uncertainty in learning.",
        "evidence": (
            "Bayesian methods for machine learning have been widely investigated, yielding principled "
            "methods for incorporating prior information into inference algorithms. They also provide "
            "an elegant approach to action selection in the face of uncertainty."
        ),
    },
    {
        "id": "expected_neutral_generalization",
        "claim": "Bayesian reinforcement learning solves the exploration-exploitation tradeoff optimally in all settings.",
        "evidence": (
            "Bayesian reinforcement learning offers a coherent probabilistic model for reinforcement learning. "
            "It provides a principled framework to express the classic exploration-exploitation dilemma."
        ),
    },
    {
        "id": "expected_contradiction",
        "claim": "The study reports that Bayesian methods are unsuitable for handling uncertainty.",
        "evidence": (
            "Bayesian reinforcement learning offers a coherent probabilistic model for reinforcement learning. "
            "It provides a principled framework for handling uncertainty in sequential decision making."
        ),
    },
    {
        "id": "multi_sentence_context_mix",
        "claim": "The posterior over model parameters is used to select actions in model-based Bayesian reinforcement learning.",
        "evidence": (
            "Bayesian methods for machine learning have been widely investigated. "
            "Model-based Bayesian Reinforcement Learning explicitly maintains a posterior over model parameters. "
            "Actions can be selected both for exploration and exploitation."
        ),
    },
]


@dataclass
class PairExample:
    name: str
    claim: str
    evidence: str


def _sentencize(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _token_overlap(claim: str, evidence: str) -> float:
    claim_tokens = set(re.findall(r"\b\w+\b", claim.lower()))
    evidence_tokens = set(re.findall(r"\b\w+\b", evidence.lower()))
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def _load_examples(path: Path) -> list[PairExample]:
    examples: list[PairExample] = []
    for idx, line in enumerate(path.read_text().splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        claim = str(payload["claim"]).strip()
        evidence_value = payload.get("evidence", "")
        if isinstance(evidence_value, list):
            evidence = "\n".join(str(item) for item in evidence_value if str(item).strip())
        else:
            evidence = str(evidence_value)
        examples.append(
            PairExample(
                name=str(payload.get("id") or payload.get("name") or f"example_{idx}"),
                claim=claim,
                evidence=evidence.strip(),
            )
        )
    return examples


def _build_examples(args: argparse.Namespace) -> list[PairExample]:
    examples: list[PairExample] = []
    if args.file:
        examples.extend(_load_examples(args.file))
    if args.claim and args.evidence:
        examples.append(PairExample(name="cli_example", claim=args.claim.strip(), evidence=args.evidence.strip()))
    if not examples:
        examples = [
            PairExample(name=item["id"], claim=item["claim"], evidence=item["evidence"])
            for item in DEFAULT_EXAMPLES
        ]
    return examples


def _score_text(model: NLIModel, claim: str, evidence: str, *, reverse: bool = False) -> dict[str, float]:
    if reverse:
        return model.classify(evidence, claim)
    return model.classify(claim, evidence)


def _margin_text(scores: dict[str, float]) -> str:
    entail_gap = scores["entails"] - GRAPH_CONFIG.nli_threshold
    contradict_gap = scores["contradicts"] - GRAPH_CONFIG.nli_contradiction_threshold
    return (
        f"entail-gap={entail_gap:+.3f} vs threshold {GRAPH_CONFIG.nli_threshold:.2f} | "
        f"contradict-gap={contradict_gap:+.3f} vs threshold {GRAPH_CONFIG.nli_contradiction_threshold:.2f}"
    )


def _print_header(examples: list[PairExample]) -> None:
    print("NLI repro / ambiguity diagnosis")
    print("=" * 80)
    print(f"Configured model                 : {GRAPH_CONFIG.nli_model_id}")
    print(f"Configured entailment threshold  : {GRAPH_CONFIG.nli_threshold:.2f}")
    print(f"Configured contradiction thresh. : {GRAPH_CONFIG.nli_contradiction_threshold:.2f}")
    print(f"Examples                         : {len(examples)}")
    print()


def _summarize_example(
    model: NLIModel,
    example: PairExample,
    top_sentences: int,
    compare_reverse: bool,
) -> dict[str, Any]:
    chunk_scores = _score_text(model, example.claim, example.evidence)
    router_result = nli_verify(example.claim, [example.evidence])
    sentences = _sentencize(example.evidence)
    sentence_scores = [
        {"text": sent, "scores": _score_text(model, example.claim, sent)}
        for sent in sentences
    ]
    sentence_scores_by_entail = sorted(
        sentence_scores,
        key=lambda item: (
            item["scores"]["entails"],
            -item["scores"]["neutral"],
            -item["scores"]["contradicts"],
        ),
        reverse=True,
    )
    sentence_scores_by_contradict = sorted(
        sentence_scores,
        key=lambda item: item["scores"]["contradicts"],
        reverse=True,
    )

    print("-" * 80)
    print(f"Example       : {example.name}")
    print(f"Claim         : {example.claim}")
    print(f"Evidence chars: {len(example.evidence)}")
    print(f"Sentences     : {len(sentences)}")
    print(f"Token overlap : {_token_overlap(example.claim, example.evidence):.3f}")
    print()
    print("Chunk-level scores")
    print(
        f"  entail={chunk_scores['entails']:.3f} | contradict={chunk_scores['contradicts']:.3f} | "
        f"neutral={chunk_scores['neutral']:.3f}"
    )
    print(f"  {_margin_text(chunk_scores)}")
    print(f"  router verdict={router_result['verdict']} | reason={router_result.get('reason')}")
    if compare_reverse:
        reverse_scores = _score_text(model, example.claim, example.evidence, reverse=True)
        print(
            "  reversed pairs "
            f"entail={reverse_scores['entails']:.3f} | contradict={reverse_scores['contradicts']:.3f} | "
            f"neutral={reverse_scores['neutral']:.3f}"
        )
    print()

    if sentence_scores:
        best = sentence_scores_by_entail[0]
        most_contradictory = sentence_scores_by_contradict[0]
        print("Best sentence by entailment")
        print(
            f"  entail={best['scores']['entails']:.3f} | contradict={best['scores']['contradicts']:.3f} | "
            f"neutral={best['scores']['neutral']:.3f}"
        )
        print(f"  {_margin_text(best['scores'])}")
        print(f"  text={best['text']}")
        print()

        print("Most contradictory sentence")
        print(
            f"  entail={most_contradictory['scores']['entails']:.3f} | "
            f"contradict={most_contradictory['scores']['contradicts']:.3f} | "
            f"neutral={most_contradictory['scores']['neutral']:.3f}"
        )
        print(f"  text={most_contradictory['text']}")
        print()

        print(f"Top {min(top_sentences, len(sentence_scores_by_entail))} sentence scores by entailment")
        for idx, item in enumerate(sentence_scores_by_entail[:top_sentences], 1):
            scores = item["scores"]
            print(
                f"  [{idx}] entail={scores['entails']:.3f} | contradict={scores['contradicts']:.3f} | "
                f"neutral={scores['neutral']:.3f}"
            )
            print(f"      {item['text']}")
        print()

    return {
        "name": example.name,
        "router_verdict": router_result["verdict"],
        "chunk_entails": chunk_scores["entails"],
        "chunk_contradicts": chunk_scores["contradicts"],
        "chunk_neutral": chunk_scores["neutral"],
        "best_sentence_entails": sentence_scores_by_entail[0]["scores"]["entails"] if sentence_scores else None,
        "best_sentence_neutral": sentence_scores_by_entail[0]["scores"]["neutral"] if sentence_scores else None,
        "sentence_count": len(sentences),
        "token_overlap": _token_overlap(example.claim, example.evidence),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact NLI verifier used by the judge on claim/evidence pairs and "
            "compare chunk-level scores against sentence-level scores to diagnose why "
            "claims escalate as Neutral."
        )
    )
    parser.add_argument("--claim", help="Single claim to score.")
    parser.add_argument("--evidence", help="Evidence text for --claim.")
    parser.add_argument(
        "--file",
        type=Path,
        help=(
            "JSONL file with one object per line: "
            '{"id": "...", "claim": "...", "evidence": "..."}'
        ),
    )
    parser.add_argument(
        "--top-sentences",
        type=int,
        default=3,
        help="How many evidence sentences to print per example.",
    )
    parser.add_argument(
        "--json-summary",
        action="store_true",
        help="Print a compact JSON summary at the end for scripting.",
    )
    parser.add_argument(
        "--compare-reverse",
        action="store_true",
        help="Also score the reversed input order to expose premise/hypothesis sensitivity.",
    )
    args = parser.parse_args()

    if bool(args.claim) != bool(args.evidence):
        parser.error("--claim and --evidence must be provided together.")

    examples = _build_examples(args)
    _print_header(examples)

    model = NLIModel.get()
    summary = [
        _summarize_example(model, example, args.top_sentences, args.compare_reverse)
        for example in examples
    ]

    if args.json_summary:
        print("=" * 80)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
