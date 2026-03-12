from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.decomposer import DecomposerAgent


@dataclass(frozen=True)
class QueryScenario:
    name: str
    query: str
    expected_split: bool


SCENARIOS: list[QueryScenario] = [
    QueryScenario(
        name="single_fact",
        query="What is reinforcement learning?",
        expected_split=False,
    ),
    QueryScenario(
        name="concept_and_application",
        query="What is quantum computing and how is it used in cryptography?",
        expected_split=True,
    ),
    QueryScenario(
        name="comparison_and_recommendation",
        query="What is the difference between dense and sparse retrieval and which one is better suited for RAG?",
        expected_split=True,
    ),
    QueryScenario(
        name="two_distinct_questions",
        query="How does BERT work and what is GPT-3?",
        expected_split=True,
    ),
    QueryScenario(
        name="pros_and_cons",
        query="What are the advantages and disadvantages of federated learning?",
        expected_split=True,
    ),
    QueryScenario(
        name="focused_explanation",
        query="Explain how attention mechanism works in transformers.",
        expected_split=False,
    ),
    QueryScenario(
        name="history_and_definition",
        query="What is PageRank and who invented it?",
        expected_split=True,
    ),
    QueryScenario(
        name="short_query",
        query="BERT vs GPT",
        expected_split=False,
    ),
]


def run() -> None:
    agent = DecomposerAgent()

    print("Agent 1 Decomposer Test")
    print("=" * 80)

    for index, scenario in enumerate(SCENARIOS, start=1):
        sub_queries = agent.decompose(scenario.query)
        actual_split = len(sub_queries) > 1
        status = "PASS" if actual_split == scenario.expected_split else "CHECK"

        print(f"\n[{index}] {scenario.name} [{status}]")
        print(f"Query: {scenario.query}")
        print(f"Expected split: {scenario.expected_split}")
        print(f"Actual split:   {actual_split}")
        print("Sub-queries:")

        for sub_index, sub_query in enumerate(sub_queries, start=1):
            print(f"  {sub_index}. {sub_query}")


if __name__ == "__main__":
    run()
