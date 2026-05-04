from __future__ import annotations
import argparse
import json
import time
from pathlib import Path


TEST_PAIRS = [
    # --- Clearly supported: near-verbatim ---
    (
        "BERT achieves state-of-the-art results on GLUE.",
        "We present BERT which obtains new state-of-the-art results on eleven NLP tasks, "
        "including pushing the GLUE score to 80.5%, MultiNLI accuracy to 86.7%.",
        "E",
    ),
    # --- Supported: paraphrase (THE HARD CASE) ---
    (
        "Attention mechanisms improve neural machine translation.",
        "In this work we propose the Transformer, a model architecture relying entirely on "
        "an attention mechanism to draw global dependencies. The Transformer reaches a new "
        "state of the art in translation quality after being trained for as little as twelve hours.",
        "E",
    ),
    # --- Supported: numbers / atomic facts ---
    (
        "GPT-3 has 175 billion parameters.",
        "GPT-3 is an autoregressive language model with 175 billion parameters trained by OpenAI "
        "on a large corpus of internet text using unsupervised pre-training.",
        "E",
    ),
    # --- Supported: indirect / inferential ---
    (
        "Transformer-based models outperform RNNs on NLP benchmarks.",
        "BERT, a deep bidirectional transformer, substantially outperforms previous systems "
        "based on LSTM and GRU architectures across all eleven NLP evaluation tasks.",
        "E",
    ),
    # --- Weakly supported: related topic, claim not directly stated ---
    (
        "Large language models struggle with mathematical reasoning.",
        "GPT-3 shows near human-level performance on some tasks, while mathematical reasoning "
        "tasks remain significantly more challenging compared to reading comprehension.",
        "N",
    ),
    # --- Neutral: same domain, unrelated claim ---
    (
        "Dropout prevents overfitting in deep neural networks.",
        "Batch normalization normalizes layer inputs by fixing the mean and variance. "
        "It allows much higher learning rates and in some cases eliminates the need for Dropout.",
        "N",
    ),
    # --- Neutral: topically adjacent but claim not supported ---
    (
        "Fine-tuning on downstream tasks always improves performance.",
        "We show that pre-training on large corpora provides useful general representations, "
        "though the benefit of fine-tuning varies substantially across tasks and domains.",
        "N",
    ),
    # --- Contradiction: clear factual negation ---
    (
        "The model was evaluated without any fine-tuning.",
        "All models were fine-tuned on the downstream task for 3 epochs with a learning rate "
        "of 2e-5 and a batch size of 32 before final evaluation on the held-out test set.",
        "C",
    ),
    # --- Contradiction: numerical ---
    (
        "The system achieves 95% accuracy on the benchmark.",
        "Our method obtains 78.3% accuracy on the held-out test set, representing a 2.1% "
        "improvement over the previous state-of-the-art baseline.",
        "C",
    ),
    # --- Contradiction: subtle / negation ---
    (
        "The proposed method requires labelled training data.",
        "Our approach is fully unsupervised and requires no labelled examples at any stage "
        "of training or evaluation.",
        "C",
    ),
]

LABELS = [
    "SUPP-verbatim", "SUPP-paraphrase", "SUPP-numbers", "SUPP-inferential",
    "WEAK", "NEUTRAL-domain", "NEUTRAL-adjacent",
    "CONTRA-clear", "CONTRA-numbers", "CONTRA-subtle",
]

MODELS = [
    {
        "id": "cross-encoder/nli-deberta-v3-small",
        "tag": "current",
        "note": "Current model in use — fast but struggles with paraphrase",
    },
    {
        "id": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "tag": "moritz-mdeberta-base",
        "note": "Multilingual DeBERTa-base, 33 NLI datasets — set in .env",
    },
    {
        "id": "cross-encoder/nli-roberta-base",
        "tag": "roberta-cross-enc",
        "note": "RoBERTa cross-encoder for NLI — literature standard",
    },
    {
        "id": "sileod/deberta-v3-small-tasksource-nli",
        "tag": "sileod-multitask",
        "note": "DeBERTa-small trained on 500+ tasks including scientific NLI",
    },
]

# Thresholds to test
ENTAIL_THRESHOLDS = [0.25, 0.30, 0.40, 0.50]
CONTRA_THRESHOLD = 0.75

# From your logs: 49 claims, 71.9s total judge time when 47/49 go to LLM
N_CLAIMS_REAL = 49
LLM_TIME_PER_CALL = 71.9 / 47


def normalize_scores(raw: list[dict]) -> dict[str, float]:
    scores = {}
    for item in raw:
        label = str(item["label"]).lower().strip()
        score = float(item["score"])
        if label in ("entailment", "entails", "support", "supported"):
            scores["entailment"] = score
        elif label in ("contradiction", "contradicts", "contradictory"):
            scores["contradiction"] = score
        elif label in ("neutral",):
            scores["neutral"] = score
        else:
            # fallback: keep raw label
            scores[label] = score
    return scores


def benchmark_model(model_id: str, tag: str, note: str, device: str) -> dict:
    from transformers import pipeline

    print(f"\n{'='*70}")
    print(f"  [{tag}] {model_id}")
    print(f"  {note}")
    print(f"{'='*70}")

    try:
        device_int = 0 if device == "cuda" else -1
        pipe = pipeline(
            "text-classification",
            model=model_id,
            device=device_int,
            top_k=None,
        )
    except Exception as ex:
        print(f"  LOAD FAILED: {ex}")
        return {"model_id": model_id, "tag": tag, "error": str(ex)}

    # Warmup
    try:
        pipe({"text": "test", "text_pair": "test"}, truncation=True, max_length=512)
    except Exception:
        pass

    # Run inference
    t0 = time.perf_counter()
    raw_results = []
    for claim, evidence, _ in TEST_PAIRS:
        r = pipe({"text": evidence, "text_pair": claim}, truncation=True, max_length=512)
        if r and isinstance(r[0], list):
            r = r[0]
        raw_results.append(r)
    elapsed = time.perf_counter() - t0
    ms_per_call = elapsed / len(TEST_PAIRS) * 1000

    print(f"\n  Speed: {ms_per_call:.0f}ms/call  |  {len(TEST_PAIRS)} pairs total: {elapsed:.2f}s")
    print(f"  Projected for {N_CLAIMS_REAL} real claims (NLI only): {N_CLAIMS_REAL * ms_per_call / 1000:.1f}s\n")

    # Show raw label names once
    sample_scores = normalize_scores(raw_results[0])
    print(f"  Raw label names: {[str(x['label']) for x in raw_results[0]]}")
    print(f"  Normalized:      {sample_scores}\n")

    threshold_results = {}

    for thresh in ENTAIL_THRESHOLDS:
        correct = 0
        n_nli_resolved = 0
        n_to_llm = 0
        detail_lines = []

        for i, (r, (claim, evidence, expected)) in enumerate(zip(raw_results, TEST_PAIRS)):
            scores = normalize_scores(r)
            e = scores.get("entailment", 0.0)
            c = scores.get("contradiction", 0.0)

            if e >= thresh:
                verdict = "E"
                n_nli_resolved += 1
            elif c >= CONTRA_THRESHOLD:
                verdict = "C"
                n_nli_resolved += 1
            else:
                verdict = "N"
                n_to_llm += 1

            ok = verdict == expected
            if ok:
                correct += 1

            status = "OK" if ok else "XX"
            detail_lines.append(
                f"    [{status}] {LABELS[i]:<22} E={e:.3f} C={c:.3f} -> {verdict}  (expected {expected})"
            )

        accuracy = correct / len(TEST_PAIRS)
        # Only count NLI-resolved claims that are CORRECT as actual savings
        # Wrong NLI decisions cost more (false positive supported = missed error)
        pct_saved = n_nli_resolved / N_CLAIMS_REAL * 100
        nli_cost = N_CLAIMS_REAL * ms_per_call / 1000
        llm_cost = n_to_llm * LLM_TIME_PER_CALL
        est_total = nli_cost + llm_cost

        print(f"  --- Threshold E>={thresh:.2f} ---")
        for line in detail_lines:
            print(line)
        print(f"  Accuracy:     {correct}/{len(TEST_PAIRS)} ({accuracy*100:.0f}%)")
        print(f"  NLI resolved: {n_nli_resolved}/{N_CLAIMS_REAL} ({pct_saved:.0f}% skip LLM)")
        print(f"  Est. time:    NLI={nli_cost:.1f}s + LLM={llm_cost:.1f}s = {est_total:.1f}s  (baseline 71.9s)")
        print(f"  Est. speedup: {71.9 / est_total:.1f}x\n")

        threshold_results[str(thresh)] = {
            "accuracy": accuracy,
            "correct": correct,
            "n_nli_resolved": n_nli_resolved,
            "n_to_llm": n_to_llm,
            "pct_saved": pct_saved,
            "est_total_s": est_total,
            "speedup": 71.9 / est_total,
        }

    return {
        "model_id": model_id,
        "tag": tag,
        "ms_per_call": ms_per_call,
        "device": device,
        "thresholds": threshold_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", default="nli_benchmark_results.json")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Subset of model tags to run (default: all). E.g. --models current sileod-multitask",
    )
    args = parser.parse_args()

    selected_tags = set(args.models) if args.models else None
    models_to_run = [m for m in MODELS if selected_tags is None or m["tag"] in selected_tags]

    print(f"Running NLI benchmark on {len(models_to_run)} model(s) | device={args.device}")
    print(f"Test set: {len(TEST_PAIRS)} pairs | Baseline judge time: 71.9s (47/49 to LLM)\n")

    all_results = []
    for m in models_to_run:
        result = benchmark_model(m["id"], m["tag"], m["note"], args.device)
        all_results.append(result)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path.resolve()}")

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY (threshold=0.40)")
    print(f"{'Model':<30} {'Acc':>6} {'%Saved':>8} {'Est.Time':>10} {'Speedup':>8}")
    print("-"*70)
    for r in all_results:
        if "error" in r:
            print(f"{r['tag']:<30} LOAD FAILED")
            continue
        t = r["thresholds"].get("0.4", {})
        print(
            f"{r['tag']:<30} {t.get('accuracy',0)*100:>5.0f}% "
            f"{t.get('pct_saved',0):>7.0f}% "
            f"{t.get('est_total_s',0):>9.1f}s "
            f"{t.get('speedup',0):>7.1f}x"
        )
    print("="*70)


if __name__ == "__main__":
    main()
