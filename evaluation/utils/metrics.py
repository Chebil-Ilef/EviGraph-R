from __future__ import annotations


METRIC_NAMES = [
    "answer_relevancy",
    "contextual_relevancy",
    "faithfulness",
    "claim_coverage",
    "attribution_faithfulness",
]

DEEPEVAL_METRIC_TO_KEY = {
    "answer relevancy": "answer_relevancy",
    "answer_relevancy": "answer_relevancy",
    "contextual relevancy": "contextual_relevancy",
    "contextual_relevancy": "contextual_relevancy",
    "faithfulness": "faithfulness",
    "claim coverage": "claim_coverage",
    "claim_coverage": "claim_coverage",
    "attribution faithfulness": "attribution_faithfulness",
    "attribution_faithfulness": "attribution_faithfulness",
}


def build_metrics(judge_model):
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.test_case import SingleTurnParams

    answer_relevancy = AnswerRelevancyMetric(
        model=judge_model,
        include_reason=False,
        async_mode=False,
    )
    contextual_relevancy = ContextualRelevancyMetric(
        model=judge_model,
        include_reason=False,
        async_mode=False,
    )
    faithfulness = FaithfulnessMetric(
        model=judge_model,
        include_reason=False,
        async_mode=False,
    )

    claim_coverage = GEval(
        name="Claim Coverage",
        model=judge_model,
        evaluation_steps=[
            "Extract the key factual claims from the expected_output.",
            "For each claim, check whether the actual_output contains that "
            "information, even if worded differently or sourced from different evidence.",
            "Score 1.0 if all key claims are present, 0.0 if none are covered. "
            "Assign proportional scores for partial coverage.",
            "Do not penalise the actual_output for containing additional correct "
            "information beyond what the expected_output states.",
        ],
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        async_mode=False,
    )

    attribution_faithfulness = GEval(
        name="Attribution Faithfulness",
        model=judge_model,
        evaluation_steps=[
            "Extract each sentence from the actual_output that carries a citation "
            "(identified by chunk_id, doc_id, or inline reference marker such as [1], [2]).",
            "For each cited sentence, locate the corresponding chunk text in the "
            "retrieval_context using the cited identifier.",
            "Check whether that specific chunk provides direct factual support "
            "for the claim made in that sentence.",
            "Penalise cases where: (a) the cited chunk does not support the sentence "
            "it is attached to, or (b) a sentence making a specific factual claim "
            "carries no citation at all.",
            "Score 1.0 if all cited sentences are correctly supported by their cited "
            "chunks, 0.0 if none are.",
        ],
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        async_mode=False,
    )

    return {
        "answer_relevancy": answer_relevancy,
        "contextual_relevancy": contextual_relevancy,
        "faithfulness": faithfulness,
        "claim_coverage": claim_coverage,
        "attribution_faithfulness": attribution_faithfulness,
    }


def evaluate_cases(test_cases, metrics):
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.evaluate.configs import (
        AsyncConfig,
        CacheConfig,
        DisplayConfig,
        ErrorConfig,
    )

    return deepeval_evaluate(
        test_cases=test_cases,
        metrics=list(metrics.values()),
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            inspect_after_run=False,
        ),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        error_config=ErrorConfig(ignore_errors=True, skip_on_missing_params=False),
    )


def normalise_metric_key(name: str) -> str:
    import re
    # Strip deepeval suffixes like " [GEval]", " [Custom]" before matching
    name = re.sub(r"\s*\[.*?\]\s*$", "", name).strip()
    lowered = name.lower().replace("-", " ")
    lowered = " ".join(lowered.split())
    return DEEPEVAL_METRIC_TO_KEY.get(lowered, lowered.replace(" ", "_"))


def result_scores(test_result) -> tuple[dict[str, float | None], dict[str, str]]:
    scores: dict[str, float | None] = {name: None for name in METRIC_NAMES}
    errors: dict[str, str] = {}

    for metric_data in test_result.metrics_data or []:
        metric_key = normalise_metric_key(metric_data.name)
        if metric_key not in scores:
            continue
        scores[metric_key] = metric_data.score
        if metric_data.error:
            errors[metric_key] = metric_data.error

    return scores, errors


def aggregate_scores(records: list[dict], variant: str) -> dict:
    from collections import defaultdict

    cat_scores: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    all_scores: dict[str, list[float]] = defaultdict(list)
    latencies: list[float] = []
    counts_by_category: dict[int, int] = defaultdict(int)
    scored_count = 0

    answer_model = records[0].get("answer_model", "") if records else ""

    for r in records:
        try:
            cat = int(r.get("category", 0))
        except (TypeError, ValueError):
            cat = 0
        counts_by_category[cat] += 1

        lat = r.get("latency_s")
        if lat is not None:
            latencies.append(lat)

        has_score = False
        for metric, val in r.get("scores", {}).items():
            if val is None:
                continue
            val = float(val)
            cat_scores[cat][metric].append(val)
            all_scores[metric].append(val)
            has_score = True
        if has_score:
            scored_count += 1

    by_category: dict[int, dict[str, float]] = {}
    for cat in sorted(counts_by_category):
        metric_lists = cat_scores.get(cat, {})
        by_category[cat] = {
            m: round(sum(v) / len(v), 4) if v else 0.0
            for m, v in metric_lists.items()
        }
        for metric in METRIC_NAMES:
            by_category[cat].setdefault(metric, 0.0)
        metric_values = [by_category[cat][m] for m in METRIC_NAMES]
        by_category[cat]["mean_5"] = round(sum(metric_values) / len(metric_values), 4)
        by_category[cat]["n"] = counts_by_category[cat]

    overall = {
        m: round(sum(v) / len(v), 4) if v else 0.0
        for m, v in all_scores.items()
    }
    for metric in METRIC_NAMES:
        overall.setdefault(metric, 0.0)
    metric_values = [overall[m] for m in METRIC_NAMES]
    overall["mean_5"] = round(sum(metric_values) / len(metric_values), 4)

    return {
        "variant": variant,
        "answer_model": answer_model,
        "by_category": by_category,
        "overall": overall,
        "latency_mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "n_records": len(records),
        "n_scored": scored_count,
    }
