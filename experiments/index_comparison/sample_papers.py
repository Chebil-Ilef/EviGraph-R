from __future__ import annotations
import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)



CLUSTERS: dict[str, dict[str, list[str]]] = {
    "CS": {
        "LLM & prompt engineering": [
            "large language model",
            "prompt engineering",
            "chain-of-thought",
            "in-context learning",
            "instruction tuning",
            "gpt-",
            "llama",
            "chatgpt",
            "language model fine",
            "few-shot prompting",
            "zero-shot prompting",
            "retrieval-augmented generation",
        ],
        "Databases & information retrieval": [
            "information retrieval",
            "dense retrieval",
            "passage retrieval",
            "document retrieval",
            "question answering retrieval",
            "bi-encoder",
            "cross-encoder",
            "bm25",
            "inverted index",
            "faiss",
            "vector database",
            "approximate nearest neighbor",
            "knowledge graph retrieval",
            "open-domain qa",
        ],
        "AI for healthcare": [
            "medical imaging",
            "clinical nlp",
            "electronic health record",
            "disease prediction",
            "drug discovery",
            "biomedical",
            "radiology",
            "pathology",
            "genomics",
            "clinical trial",
            "patient outcome",
            "medical diagnosis",
            "health informatics",
        ],
    },
    "Physics": {
        "Quantum computing": [
            "quantum circuit",
            "qubit",
            "quantum gate",
            "quantum error correction",
            "variational quantum",
            "quantum supremacy",
            "quantum advantage",
            "quantum algorithm",
            "quantum entanglement",
            "quantum coherence",
        ],
        "Astrophysics & cosmology": [
            "dark matter",
            "dark energy",
            "cosmic microwave background",
            "galaxy formation",
            "black hole",
            "neutron star",
            "gravitational wave",
            "exoplanet",
            "stellar evolution",
            "cosmological simulation",
            "redshift survey",
        ],
        "Condensed matter": [
            "superconductivity",
            "topological insulator",
            "band structure",
            "density functional theory",
            "phase transition",
            "spin-orbit coupling",
            "strongly correlated",
            "magnetic ordering",
            "lattice model",
            "ferroelectric",
        ],
    },
    "Math": {
        "Number theory": [
            "prime number",
            "diophantine equation",
            "elliptic curve",
            "modular form",
            "l-function",
            "riemann zeta",
            "arithmetic progression",
            "algebraic number field",
            "galois group",
            "p-adic",
        ],
        "Differential geometry": [
            "riemannian manifold",
            "curvature",
            "geodesic",
            "differential form",
            "symplectic geometry",
            "kahler manifold",
            "ricci flow",
            "connections on bundles",
            "lie group",
            "smooth manifold",
        ],
        "Combinatorics": [
            "graph coloring",
            "ramsey theory",
            "extremal graph",
            "chromatic polynomial",
            "hypergraph",
            "probabilistic method",
            "combinatorial identity",
            "poset",
            "matroid",
            "incidence geometry",
        ],
    },
}


def _extract_text(doc: dict) -> str:
    nc_raw = doc.get("_node_content", "{}")
    nc = json.loads(nc_raw) if isinstance(nc_raw, str) else nc_raw
    text: str = nc.get("text", "")
    # strip the leading "Title. {'section': 'Abstract', 'text': '...'}" wrapper
    m = re.search(r"'text':\s*['\"](.+)", text, re.DOTALL)
    if m:
        return m.group(1).rstrip("'\"}")
    return text


def _assign_cluster(title: str, abstract: str) -> Optional[tuple[str, str]]:
    haystack = (title + " " + abstract).lower()
    for domain, clusters in CLUSTERS.items():
        for cluster_name, keywords in clusters.items():
            if any(kw.lower() in haystack for kw in keywords):
                return domain, cluster_name
    return None


def sample_papers(
    corpus_path: str,
    per_cluster: int,
    output_path: str,
    seed: int = 42,
) -> None:
    random.seed(seed)

    # bucket: (domain, cluster) -> list of candidate docs
    buckets: dict[tuple[str, str], list[dict]] = {
        (domain, cluster): []
        for domain, clusters in CLUSTERS.items()
        for cluster in clusters
    }

    logger.info("Streaming %s …", corpus_path)
    seen_paper_ids: set[str] = set()

    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if i % 500_000 == 0 and i > 0:
                logger.info("  scanned %d docs …", i)

            doc = json.loads(line)
            paper_id = doc.get("paper_id", "")
            if not paper_id or paper_id in seen_paper_ids:
                continue

            title = doc.get("title", "")
            abstract = _extract_text(doc)

            if len(abstract.split()) < 150:
                continue

            result = _assign_cluster(title, abstract)
            if result is None:
                continue

            domain, cluster = result
            key = (domain, cluster)
            # collect up to 10× per_cluster candidates then sub-sample later
            if len(buckets[key]) < per_cluster * 10:
                buckets[key].append(
                    {
                        "paper_id": paper_id,
                        "title": title,
                        "domain": domain,
                        "cluster": cluster,
                        "abstract": abstract,
                    }
                )
                seen_paper_ids.add(paper_id)

            # stop early if all buckets already have enough candidates
            if all(len(v) >= per_cluster * 10 for v in buckets.values()):
                logger.info("All buckets full at doc %d — stopping early.", i)
                break

    logger.info("Sampling %d papers per cluster …", per_cluster)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(output, "w") as out:
        for (domain, cluster), candidates in sorted(buckets.items()):
            chosen = random.sample(candidates, min(per_cluster, len(candidates)))
            for doc in chosen:
                out.write(json.dumps(doc) + "\n")
                total += 1
            logger.info(
                "  %s / %-35s → %d papers (from %d candidates)",
                domain, cluster, len(chosen), len(candidates),
            )

    logger.info("Wrote %d papers to %s", total, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample papers per cluster from corpus.jsonl")
    parser.add_argument(
        "--corpus",
        default="/data/horse/ws/s3811141-faiss/inbe405h-unarxive/bm25_retriever/corpus.jsonl",
    )
    parser.add_argument("--per-cluster", type=int, default=5)
    parser.add_argument(
        "--output",
        default="experiments/index_comparison/results/sampled_papers.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample_papers(
        corpus_path=args.corpus,
        per_cluster=args.per_cluster,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
