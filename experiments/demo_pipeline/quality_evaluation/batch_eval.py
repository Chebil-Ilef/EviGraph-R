"""
Batch pipeline evaluation — runs a set of queries through the full EviGraph-R
pipeline, saves the complete WorkflowState as JSON and a per-stage scorecard,
and produces one graph visualisation folder per run.

Usage (from repo root):
    uv run python experiments/quality_benchmark/batch_eval.py
    uv run python experiments/quality_benchmark/batch_eval.py --queries-file my_queries.txt
    uv run python experiments/quality_benchmark/batch_eval.py --top-k 15 --no-graph-output

Output layout:
    experiments/quality_benchmark/_data/
        {YYYYMMDD_HHMMSS}/           ← one folder per batch run
            summary.json             ← per-query scorecards
            {query_slug}/
                state.json           ← full serialised WorkflowState
                scorecard.json       ← extracted quality metrics
                graph/               ← graph.html + graph.graphml (if enabled)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# ── path bootstrap ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import utils.llm  # noqa: F401  (side-effect: registers LLM env)
from config.settings import DEFAULT_EMBEDDING_MODEL
from agents.answer_generator import AnswerGeneratorAgent
from agents.decomposer import DecomposerAgent
from agents.evidence_graph_builder import EvidenceGraphBuilderAgent
from agents.judge import JudgeAgent
from retrieval.embedder import Embedder
from retrieval.retriever import HybridQueryRetriever
from schemas.objects import EvidenceGraph
from schemas.state import WorkflowState
from utils.llm import get_llm_client
from utils.qdrant import ensure_qdrant_runtime
from workflow.graph import WorkflowServices, build_workflow_graph

from dotenv import load_dotenv
load_dotenv()

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s  %(name)s  %(message)s")
for _mod in ["agents", "retrieval", "utils.graph", "workflow"]:
    logging.getLogger(_mod).setLevel(logging.DEBUG)
for _noisy in ["httpcore", "httpx", "urllib3", "transformers",
               "sentence_transformers", "huggingface_hub", "langsmith", "langchain"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ── default queries ───────────────────────────────────────────────────────────
DEFAULT_QUERIES = [
    "top quark production and decay at the LHC?",
    "What is contrastive learning and how is it evaluated in out-of-domain retrieval?",
    "How does attention mechanism work in transformers?",
    "What is the difference between dense and sparse retrieval and which is better for RAG?",
    "How are graph neural networks applied to molecular property prediction?",
    "What methods are used for protein structure prediction?",
    "How does federated learning preserve privacy?",
    "What is reinforcement learning from human feedback (RLHF)?",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _slug(query: str, max_len: int = 45) -> str:
    return re.sub(r"[^\w]+", "_", query.lower().strip())[:max_len].strip("_")


def _scorecard(state: WorkflowState) -> dict:
    """Extract structured quality metrics from a finished WorkflowState."""

    # --- Decomposition ---
    sub_queries = state.sub_queries or []
    decomp = {
        "n_sub_queries": len(sub_queries),
        "budget_weights": [round(sq.budget_weight, 3) for sq in sub_queries],
        "sections": [
            [s.value for s in sq.sections] for sq in sub_queries
        ],
    }

    # --- Retrieval ---
    docs = state.retrieved_documents or []
    scores = [d.score for d in docs]
    section_counts: Counter = Counter(d.section_title or "?" for d in docs)
    retrieval = {
        "n_docs": len(docs),
        "score_min": round(min(scores), 4) if scores else None,
        "score_max": round(max(scores), 4) if scores else None,
        "score_mean": round(sum(scores) / len(scores), 4) if scores else None,
        "section_distribution": dict(section_counts.most_common()),
    }

    # --- Evidence graph ---
    graph: EvidenceGraph = state.evidence_graph or EvidenceGraph()
    node_type_counts: Counter = Counter(n.node_type.value for n in graph.nodes)
    edge_type_counts: Counter = Counter(e.relation for e in graph.edges)
    n_claims = node_type_counts.get("claim", 0)
    n_concepts = node_type_counts.get("concept", 0)
    n_chunks = node_type_counts.get("chunk", 0)
    graph_metrics = {
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
        "node_types": dict(node_type_counts),
        "edge_types": dict(edge_type_counts),
        "claims_per_doc": round(n_claims / max(len(docs), 1), 2),
        "concepts_per_doc": round(n_concepts / max(len(docs), 1), 2),
        "isolated_claim_ratio": _isolated_claim_ratio(graph),
    }

    # --- Judge ---
    relations = state.judged_relations or []
    verdict_details = state.verdict_details or {}
    verdict_counts: Counter = Counter()
    verifier_counts: Counter = Counter()
    for cid, vd in verdict_details.items():
        verdict_counts[vd.get("verdict", "?")] += 1
        verifier_counts[vd.get("verifier_used", "?")] += 1
    judge_metrics = {
        "n_judged_relations": len(relations),
        "n_filtered_docs": len(state.filtered_evidence or []),
        "verdict_distribution": dict(verdict_counts),
        "verifier_distribution": dict(verifier_counts),
        "supported_ratio": round(
            verdict_counts.get("Supported", 0) / max(sum(verdict_counts.values()), 1), 3
        ),
        "escalation_to_llm": verifier_counts.get("nli->llm_judge", 0)
            + verifier_counts.get("llm_judge", 0),
    }

    # --- Answer ---
    answer = state.final_answer
    answer_metrics: dict
    if answer and answer.text:
        conflict_count = sum(1 for s in answer.sentences if s.conflict_flag)
        cited_docs = {c.doc_id for s in answer.sentences for c in s.citations}
        answer_metrics = {
            "n_sentences": len(answer.sentences),
            "n_cited_docs": len(cited_docs),
            "conflict_sentences": conflict_count,
            "answer_length_chars": len(answer.text),
            "has_answer": True,
        }
    else:
        answer_metrics = {"has_answer": False}

    return {
        "decomposition": decomp,
        "retrieval": retrieval,
        "graph": graph_metrics,
        "judge": judge_metrics,
        "answer": answer_metrics,
        "errors": state.errors,
        "n_logs": len(state.logs),
    }


def _isolated_claim_ratio(graph: EvidenceGraph) -> float:
    """Fraction of claim nodes with no edges (disconnected from the graph)."""
    claims = {n.node_id for n in graph.nodes if n.node_type.value == "claim"}
    if not claims:
        return 0.0
    connected = {e.source for e in graph.edges} | {e.target for e in graph.edges}
    isolated = claims - connected
    return round(len(isolated) / len(claims), 3)


def _state_to_json(state: WorkflowState) -> dict:
    """Serialise WorkflowState to a plain dict, handling nested Pydantic models."""
    return json.loads(state.model_dump_json())


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch pipeline quality evaluation")
    parser.add_argument(
        "--queries-file",
        type=Path,
        help="Plain-text file with one query per line (overrides built-in queries)",
    )
    parser.add_argument("--model-key", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--no-graph-output", action="store_true", help="Skip pyvis/GraphML dump"
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Run only the first N queries (useful for quick smoke tests)",
    )
    args = parser.parse_args()

    queries: list[str]
    if args.queries_file:
        queries = [
            l.strip()
            for l in args.queries_file.read_text().splitlines()
            if l.strip() and not l.startswith("#")
        ]
    else:
        queries = DEFAULT_QUERIES

    if args.limit is not None:
        queries = queries[: args.limit]

    # ── output directory ──────────────────────────────────────────────────────
    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = REPO_ROOT / "experiments" / "quality" / "_data" / batch_ts
    results_root.mkdir(parents=True, exist_ok=True)
    print(f"\nBatch output directory: {results_root}\n")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    profile = os.getenv("INDEXING_PROFILE", "local")
    print(f"Ensuring Qdrant is running (profile: {profile})...")
    ensure_qdrant_runtime(profile, startup_timeout=30)
    print("✓ Qdrant ready\n")

    # ── services (shared across all queries) ──────────────────────────────────
    llm_client = get_llm_client()
    embedder = Embedder.from_model_key(args.model_key)
    retriever = HybridQueryRetriever(model_key=args.model_key)

    all_summaries: list[dict] = []

    for idx, query in enumerate(queries, 1):
        slug = _slug(query)
        run_dir = results_root / slug
        run_dir.mkdir(parents=True, exist_ok=True)

        graph_output_dir = None if args.no_graph_output else (run_dir / "graph")

        print(f"{'═' * 70}")
        print(f"  [{idx}/{len(queries)}] {query}")
        print(f"{'═' * 70}")

        services = WorkflowServices(
            decomposer=DecomposerAgent(llm_client=llm_client),
            embedder=embedder,
            retriever=retriever,
            evidence_graph_builder=EvidenceGraphBuilderAgent(
                llm_client=llm_client,
                output_dir=graph_output_dir,
            ),
            judge=JudgeAgent(llm_client=llm_client),
            answer_generator=AnswerGeneratorAgent(llm_client=llm_client),
        )

        workflow = build_workflow_graph(services)

        t0 = time.perf_counter()
        try:
            final_state_dict = workflow.invoke(WorkflowState(query=query).model_dump())
            state = WorkflowState(**final_state_dict)
            elapsed = round(time.perf_counter() - t0, 1)
            ok = True
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 1)
            logger.error("Pipeline crashed for query %r: %s", query, exc)
            state = WorkflowState(query=query, errors=[str(exc)])
            ok = False

        # ── persist state ─────────────────────────────────────────────────────
        state_path = run_dir / "state.json"
        state_path.write_text(json.dumps(_state_to_json(state), indent=2))

        # ── scorecard ─────────────────────────────────────────────────────────
        card = _scorecard(state)
        card["query"] = query
        card["slug"] = slug
        card["elapsed_s"] = elapsed
        card["pipeline_ok"] = ok

        scorecard_path = run_dir / "scorecard.json"
        scorecard_path.write_text(json.dumps(card, indent=2))

        all_summaries.append(card)

        # ── console summary ───────────────────────────────────────────────────
        g = card["graph"]
        j = card["judge"]
        a = card["answer"]
        print(f"  elapsed : {elapsed}s")
        print(f"  retrieval: {card['retrieval']['n_docs']} docs")
        print(f"  graph   : {g['n_nodes']} nodes  {g['n_edges']} edges  "
              f"({g['node_types']})  isolated_claims={g['isolated_claim_ratio']}")
        print(f"  judge   : supported={j['verdict_distribution'].get('Supported',0)}  "
              f"verifiers={j['verifier_distribution']}")
        if a.get("has_answer"):
            print(f"  answer  : {a['n_sentences']} sentences  "
                  f"{a['n_cited_docs']} cited docs  "
                  f"conflicts={a['conflict_sentences']}")
        else:
            print("  answer  : (none)")
        if card["errors"]:
            print(f"  ERRORS  : {card['errors']}")
        print()

    # ── batch summary ─────────────────────────────────────────────────────────
    summary_path = results_root / "summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"\nSummary written to: {summary_path}")

    # brief aggregate stats
    successful = [c for c in all_summaries if c["pipeline_ok"]]
    print(f"\n{'═' * 70}")
    print(f"  Batch complete — {len(successful)}/{len(all_summaries)} queries succeeded")
    if successful:
        avg_elapsed = round(sum(c["elapsed_s"] for c in successful) / len(successful), 1)
        avg_docs = round(sum(c["retrieval"]["n_docs"] for c in successful) / len(successful), 1)
        avg_supported = round(
            sum(c["judge"]["verdict_distribution"].get("Supported", 0) for c in successful)
            / len(successful), 1
        )
        print(f"  Avg elapsed   : {avg_elapsed}s")
        print(f"  Avg retrieved : {avg_docs} docs")
        print(f"  Avg supported : {avg_supported} claims")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
