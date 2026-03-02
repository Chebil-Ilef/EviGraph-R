import argparse
import json
import os
import time
from datasets import load_dataset


def make_json_serializable(obj):
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    return obj


def log(msg: str):
    print(msg, flush=True)


def main():
    p = argparse.ArgumentParser(description="Stream unarxive_2024 and split into round-robin JSONL batches.")
    p.add_argument("--num-batches", type=int, default=12, help="Number of batch JSONL files to create")
    p.add_argument("--out-dir", type=str, default="unarxive_batches_sf", help="Output directory")
    p.add_argument("--prefix", type=str, default="batch", help="Filename prefix (batch_01.jsonl, ...)")
    p.add_argument("--dataset", type=str, default="ines-besrour/unarxive_2024", help="HF dataset name")
    p.add_argument("--split", type=str, default="train", help="Dataset split")
    p.add_argument("--max-records", type=int, default=None, help="Optional cap (omit to stream full dataset)")
    p.add_argument("--log-every", type=int, default=1000, help="Log every N records (0 disables)")
    p.add_argument("--hf-token", type=str, default=None, help="Optional HF token (or set HF_TOKEN env var)")
    args = p.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    os.makedirs(args.out_dir, exist_ok=True)

    log("Streaming dataset...")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    pad = len(str(args.num_batches))
    batch_paths = [
        os.path.join(args.out_dir, f"{args.prefix}_{str(i + 1).zfill(pad)}.jsonl")
        for i in range(args.num_batches)
    ]

    batch_files = []
    batch_counts = [0] * args.num_batches
    total_written = 0
    t0 = time.time()

    try:
        batch_files = [open(path, "w", encoding="utf-8") for path in batch_paths]

        for row in ds:
            if args.max_records is not None and total_written >= args.max_records:
                break

            batch_idx = total_written % args.num_batches
            clean_row = make_json_serializable(row)
            if not clean_row.get("jsonl", "").strip():
                continue

            batch_files[batch_idx].write(json.dumps(clean_row, ensure_ascii=False) + "\n")
            batch_counts[batch_idx] += 1
            total_written += 1

            if args.log_every and total_written % args.log_every == 0:
                dt = max(time.time() - t0, 1e-9)
                rps = total_written / dt
                log(f"  {total_written} records written... ({rps:.2f} rec/s)")

    except KeyboardInterrupt:
        log("\nInterrupted by user (Ctrl+C). Closing files cleanly...")
    finally:
        for f in batch_files:
            try:
                f.close()
            except Exception:
                pass

    dt = max(time.time() - t0, 1e-9)
    log("\nDone.")
    log(f"Total records written: {total_written}")
    log(f"Output directory: {args.out_dir}")
    if total_written:
        log(f"Average batch size: {total_written / args.num_batches:.1f} records")
        log(f"Overall throughput: {total_written / dt:.2f} rec/s")

    for i, count in enumerate(batch_counts):
        fname = os.path.basename(batch_paths[i])
        log(f"  {fname}: {count} records")

    if total_written == 0:
        log("\nWARNING: 0 records written!!")


if __name__ == "__main__":
    main()

    # uv run src/utils/batch_unarxiv.py \
    #     --num-batches 12 \
    #     --out-dir _data/unarxive_batches \
    #     --max-records 100 \
    #     --log-every 50