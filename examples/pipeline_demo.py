"""
Usage:
    uv run python pipeline_demo.py
    uv run python pipeline_demo.py --query "How does contrastive learning work?"
    uv run python pipeline_demo.py --query "..." --top-k 15 --no-graph-output

ON HPC:
    ./scripts/run_demo_pipeline_interactive.sh 
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import utils.llm  
from config.settings import DEFAULT_EMBEDDING_MODEL
from schemas.objects import FinalAnswer, EvidenceGraph, SubQuery
from schemas.state import WorkflowState
from agents.decomposer import DecomposerAgent
from agents.evidence_graph_builder import EvidenceGraphBuilderAgent
from agents.judge import JudgeAgent
from agents.answer_generator import AnswerGeneratorAgent
from retrieval.embedder import Embedder
from retrieval.retriever import HybridQueryRetriever
from workflow.graph import WorkflowServices, build_workflow_graph
from utils.llm import get_llm_client
from utils.qdrant import ensure_qdrant_runtime
import os

from dotenv import load_dotenv
load_dotenv()



SEP = "─" * 70

def _section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")

def _print_decomposition(sub_queries: list[SubQuery]) -> None:
    _section("AGENT 1 — DECOMPOSITION")
    for i, sq in enumerate(sub_queries, 1):
        sections = ", ".join(s.value for s in sq.sections) or "—"
        print(f"  [{i}] weight={sq.budget_weight:.2f}  sections=[{sections}]")
        print(f"      {sq.text}")

def _print_retrieved(docs) -> None:
    _section(f"RETRIEVAL — {len(docs)} unique chunks")
    for doc in docs:
        snippet = doc.content[:120].replace("\n", " ")
        print(f"  [{doc.score:.3f}]  {doc.doc_id}  chunk={doc.chunk_id[-12:]}")
        print(f"         section={doc.section_title or '?'}  ↳ {snippet}…")

def _print_graph(graph: EvidenceGraph) -> None:
    _section("AGENT 2 — EVIDENCE GRAPH")
    type_counts = Counter(n.node_type.value for n in graph.nodes)
    rel_counts   = Counter(e.relation for e in graph.edges)
    print(f"  Nodes ({len(graph.nodes)} total):")
    for t, c in sorted(type_counts.items()):
        print(f"    {t:<12} {c}")
    print(f"  Edges ({len(graph.edges)} total):")
    for r, c in sorted(rel_counts.items()):
        print(f"    {r:<20} {c}")

    print("\n  Sample claims extracted:")
    claims = [n for n in graph.nodes if n.node_type.value == "claim"][:5]
    if claims:
        for c in claims:
            print(f"    • {c.text[:110]}")
    else:
        print("    (none)")

def _print_judge(state: WorkflowState) -> None:
    _section("AGENT 3 — JUDGE")
    relations = state.judged_relations
    filtered = state.filtered_evidence
    print(f"  Verified edges : {len(relations)}")
    print(f"  Docs forwarded : {len(filtered)}")
    verdict_counts: Counter = Counter()
    for rel in relations:
        verdict_counts[rel.metadata.get("verifier", "?")] += 1
    for verifier, count in sorted(verdict_counts.items()):
        print(f"    {verifier:<20} {count}")

def _print_answer(state: WorkflowState) -> None:
    _section("AGENT 4 — FINAL ANSWER")
    answer = state.final_answer
    if not answer or not answer.text:
        print("  (no answer generated)")
        return
    print(f"\n  {answer.text}\n")
    if answer.sentences:
        print(f"  Citations ({len(answer.sentences)} sentences):")
        for s in answer.sentences[:5]:
            conflict = " [CONFLICT]" if s.conflict_flag else ""
            for c in s.citations:
                print(f"    • {c.doc_id}  {c.scicite_label or ''}  score={c.rel_score or 0:.2f}  verdict={c.verdict or '?'}{conflict}")

def _print_logs(state: WorkflowState) -> None:
    _section("WORKFLOW LOGS")
    for line in state.logs:
        print(f"  {line}")
    if state.errors:
        print("\n  ERRORS:")
        for e in state.errors:
            print(f"  ✗ {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="top quark production and decay at the LHC?")
    parser.add_argument("--model-key", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-graph-output", action="store_true", help="Skip pyvis/GraphML dump")
    args = parser.parse_args()

    output_dir = None if args.no_graph_output else Path("_data/graphs")

    print(f"\n{'═' * 70}")
    print(f"  EviGraph-R  —  Dev Pipeline Run")
    print(f"{'═' * 70}")
    print(f"  Query      : {args.query}")
    print(f"  Model      : {args.model_key}")
    print(f"  Top-K      : {args.top_k}")
    print(f"  Graph out  : {output_dir or 'disabled'}")

    # Ensure Qdrant is running 
    profile = os.getenv("INDEXING_PROFILE", "local")
    print(f"\nEnsuring Qdrant is running (profile: {profile})...")
    
    try:
        ensure_qdrant_runtime(profile, startup_timeout=30)
        print("✓ Qdrant is ready\n")
    except Exception as e:
        print(f"\n{'═' * 70}")
        print("  ❌ QDRANT CONNECTION FAILED")
        print(f"{'═' * 70}")
        print(f"\nError: {e}\n")
        
        # Check if we're on a login node
        import socket
        hostname = socket.gethostname()
        if "login" in hostname.lower() or "c1.capella" in hostname.lower() or "c1" == hostname:
            print("⚠️  YOU ARE ON A LOGIN/HEAD NODE")
            print("\nLogin/head nodes don't have permission to run Singularity containers.")
            print("You must run this on a COMPUTE NODE.\n")
        
        print("✅ CORRECT WAY TO RUN THIS:")
        print(f"  ./scripts/run_demo_pipeline_interactive.sh\n")
        print("This script will:")
        print("  • Automatically request a compute node")
        print("  • Start Qdrant on that node")
        print("  • Run the pipeline with your query\n")
        
        print(f"{'═' * 70}\n")
        sys.exit(1)  # EXIT - DO NOT CONTINUE

    # services 
    llm_client  = get_llm_client()
    embedder    = Embedder.from_model_key(args.model_key)
    retriever   = HybridQueryRetriever(model_key=args.model_key)

    services = WorkflowServices(
        decomposer=DecomposerAgent(llm_client=llm_client),
        embedder=embedder,
        retriever=retriever,
        evidence_graph_builder=EvidenceGraphBuilderAgent(
            llm_client=llm_client,
            output_dir=output_dir,
        ),
        judge=JudgeAgent(llm_client=llm_client),
        answer_generator=AnswerGeneratorAgent(llm_client=llm_client),
    )

    workflow = build_workflow_graph(services)
    final_state_dict = workflow.invoke(
        WorkflowState(query=args.query).model_dump()
    )
    
    final_state = WorkflowState(**final_state_dict)

    _print_decomposition(final_state.sub_queries)
    _print_retrieved(final_state.retrieved_documents)
    _print_graph(final_state.evidence_graph or EvidenceGraph())
    _print_judge(final_state)
    _print_answer(final_state)
    _print_logs(final_state)

    if output_dir and (output_dir).exists():
        html_files = list(output_dir.glob("*.html"))
        if html_files:
            _section("GRAPH VISUALIZATION")
            for f in html_files:
                print(f"  pyvis HTML  → {f}")
            
            # Check if we're on HPC based on profile
            profile = os.getenv("INDEXING_PROFILE", "local")
            
            if profile == "hpc":
                print("\n  📡 You're on HPC - use port forwarding to view:")
                print("     ./scripts/port_forwarding.sh")
                print("     Then open http://localhost:8888 in your laptop browser")
            else:
                print("  Open the HTML file in your browser to explore the graph interactively.")



if __name__ == "__main__":
    main()
