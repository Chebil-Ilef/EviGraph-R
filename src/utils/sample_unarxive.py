import argparse
import json
import os
from datasets import load_dataset

def make_json_serializable(obj):

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    return obj

def main():
    p = argparse.ArgumentParser(description="Stream a sample from unarxive_2024 and write JSONL.")
    p.add_argument("--n", type=int, default=200, help="Number of records to write")
    p.add_argument("--out", type=str, default="unarxive_2024_sample.jsonl", help="Output JSONL path")
    p.add_argument("--dataset", type=str, default="ines-besrour/unarxive_2024", help="HF dataset name")
    p.add_argument("--split", type=str, default="train", help="Split name (usually 'train')")
    p.add_argument("--hf-token", type=str, default=None, help="Optional HF token (or set HF_TOKEN env var)")
    args = p.parse_args()

    # optional: set token for higher rate limits
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            if not row.get("jsonl"):
                continue
            clean_row = make_json_serializable(row)
            f.write(json.dumps(clean_row, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.n:
                break

    print(f"Wrote {written} rows to {args.out}")

if __name__ == "__main__":
    main()
    # uv run src/utils/sample_unarxive.py --n 200 --out _data/unarxive_2024_sample_200_articles.jsonl
