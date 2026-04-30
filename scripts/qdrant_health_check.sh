#!/bin/bash
#SBATCH --job-name=qdrant-health-check
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R/logs/qdrant_health_check_%j.log

set -euo pipefail

REPO=/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R
COLLECTION=unarxive_chunks

# Use a different port so we never interfere with the running indexing job (6333)
PORT=6335
GRPC_PORT=6336

INSTANCE=evigraph-qdrant-healthcheck-${SLURM_JOB_ID:-local}

QDRANT_SIF=${REPO}/qdrant.sif
SNAPSHOTS_DIR=${REPO}/snapshots
STORAGE=${SLURM_TMPDIR:-/tmp}/${USER}/qdrant_healthcheck_${SLURM_JOB_ID:-local}
LOG=${REPO}/logs/qdrant_healthcheck_server_${SLURM_JOB_ID:-local}.log
REPORT=${REPO}/logs/qdrant_health_report_${SLURM_JOB_ID:-local}.json

mkdir -p "$STORAGE" "${REPO}/logs"

# Pick the most recent snapshot automatically
SNAPSHOT_FILE=$(ls -t "${SNAPSHOTS_DIR}/${COLLECTION}/"*.snapshot 2>/dev/null | head -1)
if [[ -z "$SNAPSHOT_FILE" ]]; then
  echo "ERROR: no .snapshot file found in ${SNAPSHOTS_DIR}/${COLLECTION}/"
  exit 1
fi
SNAPSHOT_NAME=$(basename "$SNAPSHOT_FILE")
echo "Using snapshot: $SNAPSHOT_NAME"

# Start Qdrant  
singularity instance start \
  --userns \
  --bind "${STORAGE}:/qdrant/storage" \
  --bind "${SNAPSHOTS_DIR}/${COLLECTION}:/qdrant/snapshots/${COLLECTION}" \
  --env "QDRANT__SERVICE__HTTP_PORT=${PORT}" \
  --env "QDRANT__SERVICE__GRPC_PORT=${GRPC_PORT}" \
  "$QDRANT_SIF" \
  "$INSTANCE"

singularity exec \
  --env "QDRANT__SERVICE__HTTP_PORT=${PORT}" \
  --env "QDRANT__SERVICE__GRPC_PORT=${GRPC_PORT}" \
  "instance://${INSTANCE}" /qdrant/qdrant > "$LOG" 2>&1 &
QDRANT_PID=$!

cleanup() {
  echo "Stopping Qdrant instance..."
  singularity instance stop "$INSTANCE" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for readiness 
echo "Waiting for Qdrant on port ${PORT}..."
READY=0
for i in $(seq 1 120); do
  if curl -sf "http://localhost:${PORT}/healthz" >/dev/null 2>&1; then
    READY=1
    echo "Qdrant ready after $((i * 2))s"
    break
  fi
  if ! kill -0 "$QDRANT_PID" 2>/dev/null; then
    echo "ERROR: Qdrant process died. Last log lines:"
    tail -40 "$LOG" || true
    exit 1
  fi
  sleep 2
done

if [[ $READY -eq 0 ]]; then
  echo "ERROR: Qdrant did not become ready within 240s"
  tail -40 "$LOG" || true
  exit 1
fi

# Recover snapshot
echo ""
echo "=========================================="
echo " Recovering snapshot (this may take a few minutes)..."
echo "=========================================="
RECOVER_RESP=$(curl -sS -X PUT \
  "http://localhost:${PORT}/collections/${COLLECTION}/snapshots/recover?wait=true" \
  -H 'Content-Type: application/json' \
  -d "{
    \"location\": \"file:///qdrant/snapshots/${COLLECTION}/${SNAPSHOT_NAME}\",
    \"priority\": \"snapshot\"
  }")
echo "Recovery response: $RECOVER_RESP"

RECOVER_OK=$(echo "$RECOVER_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result', False))" 2>/dev/null || echo "unknown")
if [[ "$RECOVER_OK" != "True" && "$RECOVER_OK" != "true" ]]; then
  echo "WARNING: snapshot recovery may have failed — continuing to report what's there"
fi

echo ""
echo "=========================================="
echo " QDRANT HEALTH REPORT"
echo " Node:       $(hostname)"
echo " Port:       ${PORT}  (isolated from running job on 6333)"
echo " Snapshot:   ${SNAPSHOT_NAME}"
echo " Timestamp:  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=========================================="

python3 - <<PYEOF
import json, urllib.request, datetime
from collections import Counter, defaultdict

PORT       = 6335
COLLECTION = "unarxive_chunks"
REPORT     = "${REPORT}"
SAMPLE     = 10000   # scroll this many points for statistical analysis

# ── helpers  ──────
def fetch(path):
    try:
        with urllib.request.urlopen(f"http://localhost:{PORT}{path}") as r:
            return json.load(r)
    except Exception as e:
        return {"error": str(e)}

def post(path, payload):
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"http://localhost:{PORT}{path}", data=data,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except Exception as e:
        return {"error": str(e)}

def bar(value, total, width=40):
    if total == 0:
        return "[" + "-"*width + "]"
    filled = int(width * value / total)
    return "[" + "█"*filled + "░"*(width-filled) + "]"

#  1. version  
ver = fetch("/")
print("\n" + "="*70)
print("  [1] QDRANT VERSION")
print("="*70)
print(f"  Version : {ver.get('version','?')}")
print(f"  Commit  : {ver.get('commit','?')[:12]}")

#  2. collection overview 
coll = fetch(f"/collections/{COLLECTION}")
res  = coll.get("result", {})
params  = res.get("config", {}).get("params", {})
vectors = params.get("vectors", {})
opt_cfg = res.get("config", {}).get("optimizer_config", {})
wal_cfg = res.get("config", {}).get("wal_config", {})

total_points  = res.get("points_count", 0)
indexed_vecs  = res.get("indexed_vectors_count", 0)
segments      = res.get("segments_count", 0)
status        = res.get("status", "?")
optimizer_st  = res.get("optimizer_status", "?")

print("\n" + "="*70)
print("  [2] COLLECTION OVERVIEW")
print("="*70)
print(f"  Status           : {status}")
print(f"  Optimizer status : {optimizer_st}")
print(f"  Total points     : {total_points:,}")
print(f"  Indexed vectors  : {indexed_vecs:,}")
print(f"  Segments         : {segments}")
print(f"  Shard count      : {params.get('shard_number','?')}")
print(f"  Replication      : {params.get('replication_factor','?')}")
print(f"  On-disk payload  : {params.get('on_disk_payload','?')}")
print(f"\n  WAL capacity     : {wal_cfg.get('wal_capacity_mb','?')} MB")
print(f"  Flush interval   : {opt_cfg.get('flush_interval_sec','?')}s")
print(f"  Indexing thresh  : {opt_cfg.get('indexing_threshold','?')} vectors")

print(f"\n  Vector configs:")
for vname, vcfg in vectors.items():
    print(f"    [{vname}]  size={vcfg.get('size','?')}  dist={vcfg.get('distance','?')}  on_disk={vcfg.get('on_disk','?')}")
for vname, vcfg in params.get("sparse_vectors", {}).items():
    idx = vcfg.get("index", {})
    print(f"    [{vname}] (sparse)  on_disk={idx.get('on_disk','?')}")

# Storage estimates
dense_size  = next(iter(vectors.values()), {}).get("size", 1024)
dense_gb    = total_points * dense_size * 4 / 1024**3
print(f"\n  Estimated dense vector storage : {dense_gb:.2f} GB  ({dense_size}D x float32 x {total_points:,})")

#  3. shard/cluster 
cluster = fetch(f"/collections/{COLLECTION}/cluster")
cr      = cluster.get("result", {})
print("\n" + "="*70)
print("  [3] SHARDS & CLUSTER")
print("="*70)
print(f"  Peer ID      : {cr.get('peer_id','?')}")
print(f"  Shard count  : {cr.get('shard_count','?')}")
for s in cr.get("local_shards", []):
    print(f"    shard {s['shard_id']}  points={s['points_count']:,}  state={s['state']}")

#  4. deep sample analysis 
print("\n" + "="*70)
print(f"  [4] DEEP SAMPLE ANALYSIS  (scrolling up to {SAMPLE:,} points)")
print("="*70)

unique_papers   = set()
papers_complete = {}   # arxiv_id -> total_chunks (from payload field)
papers_seen_chunks = defaultdict(int)  # arxiv_id -> chunks seen in sample
by_year         = Counter()
by_category     = Counter()
by_chunk_type   = Counter()
by_discipline   = Counter()
sparse_nnz      = []
fetched         = 0
offset          = None
batch           = 500

while fetched < SAMPLE:
    payload_req = {"limit": batch, "with_payload": True, "with_vector": False}
    if offset:
        payload_req["offset"] = offset
    resp   = post(f"/collections/{COLLECTION}/points/scroll", payload_req)
    points = resp.get("result", {}).get("points", [])
    offset = resp.get("result", {}).get("next_page_offset")
    if not points:
        break
    for p in points:
        pl = p.get("payload") or {}
        arxiv = pl.get("paper_id_arxiv", "")
        if arxiv:
            unique_papers.add(arxiv)
            papers_seen_chunks[arxiv] += 1
            tc = pl.get("total_chunks")
            if tc:
                papers_complete[arxiv] = tc
        yr = pl.get("year")
        if yr:
            by_year[int(yr)] += 1
        for cat in (pl.get("categories") or []):
            by_category[cat] += 1
        by_chunk_type[pl.get("chunk_type", "unknown")] += 1
        by_discipline[pl.get("discipline", "unknown")] += 1
    fetched += len(points)
    if not offset:
        break

print(f"  Points scrolled           : {fetched:,}")
print(f"  Unique arxiv IDs in sample: {len(unique_papers):,}")

# Use total_chunks from payload — this is the ground truth stored per point,
# not a count of how many chunks we happened to scroll past (which is biased
# by scroll order and sample size).
if papers_complete:
    tc_values = list(papers_complete.values())
    avg_chunks_true = sum(tc_values) / len(tc_values)
    min_chunks = min(tc_values)
    max_chunks = max(tc_values)
    est_papers_true = int(total_points / avg_chunks_true)
    print(f"\n  Chunks/paper stats (from total_chunks payload field, {len(tc_values):,} papers in sample):")
    print(f"    avg : {avg_chunks_true:.1f}")
    print(f"    min : {min_chunks}")
    print(f"    max : {max_chunks}")
    # distribution buckets
    buckets = Counter()
    for v in tc_values:
        if v == 1:       buckets["1"] += 1
        elif v <= 5:     buckets["2-5"] += 1
        elif v <= 10:    buckets["6-10"] += 1
        elif v <= 25:    buckets["11-25"] += 1
        elif v <= 50:    buckets["26-50"] += 1
        elif v <= 100:   buckets["51-100"] += 1
        else:            buckets["100+"] += 1
    for label in ["1","2-5","6-10","11-25","26-50","51-100","100+"]:
        cnt = buckets.get(label, 0)
        pct = cnt / len(tc_values) * 100
        print(f"    {label:<8}  {cnt:>6,}  {bar(cnt, len(tc_values), 25)}  {pct:.1f}%")
    print(f"\n  Estimated total unique papers : {est_papers_true:,}  (total_points / avg_chunks_true)")
else:
    # fallback: scroll-count ratio — label it clearly as unreliable
    if fetched > 0 and len(unique_papers) > 0:
        avg_scroll = fetched / len(unique_papers)
        print(f"  WARNING: total_chunks not found in sample; scroll-ratio estimate unreliable")
        print(f"  Scroll-ratio avg chunks/paper : {avg_scroll:.1f}  (biased low — do not trust)")

# Papers where we've seen ALL their chunks (complete in sample)
fully_covered = sum(
    1 for pid, seen in papers_seen_chunks.items()
    if pid in papers_complete and seen >= papers_complete[pid]
)
print(f"\n  Fully-covered papers in sample: {fully_covered:,}")

print(f"\n  Chunk types:")
for ct, cnt in by_chunk_type.most_common():
    pct = cnt / fetched * 100 if fetched else 0
    print(f"    {ct:<20} {cnt:>7,}  {bar(cnt, fetched, 30)}  {pct:.1f}%")

print(f"\n  Disciplines (top 10):")
for disc, cnt in by_discipline.most_common(10):
    pct = cnt / fetched * 100 if fetched else 0
    print(f"    {disc:<25} {cnt:>7,}  {pct:.1f}%")

print(f"\n  Years (top 15):")
for yr, cnt in sorted(by_year.most_common(15), key=lambda x: -x[0]):
    pct = cnt / fetched * 100 if fetched else 0
    print(f"    {yr}  {cnt:>7,}  {bar(cnt, max(by_year.values()), 30)}  {pct:.1f}%")

print(f"\n  arXiv categories (top 15):")
for cat, cnt in by_category.most_common(15):
    pct = cnt / fetched * 100 if fetched else 0
    print(f"    {cat:<12} {cnt:>7,}  {bar(cnt, max(by_category.values()), 25)}  {pct:.1f}%")

# 5. payload schema  
print("\n" + "="*70)
print("  [5] PAYLOAD SCHEMA")
print("="*70)
schema = res.get("payload_schema", {})
for field, finfo in sorted(schema.items()):
    dtype  = finfo.get("data_type", "?") if isinstance(finfo, dict) else str(finfo)
    npts   = finfo.get("points", 0) if isinstance(finfo, dict) else 0
    cov    = f"{npts/total_points*100:.1f}%" if total_points and npts else "?"
    print(f"    {field:<22} type={dtype:<12} coverage={cov}  ({npts:,} pts)")

# 6. sample payload 
print("\n" + "="*70)
print("  [6] SAMPLE POINT PAYLOAD")
print("="*70)
scroll1 = post(f"/collections/{COLLECTION}/points/scroll",
               {"limit": 1, "with_payload": True, "with_vector": False})
pts = scroll1.get("result", {}).get("points", [])
if pts:
    pl = pts[0].get("payload", {})
    for k in sorted(pl.keys()):
        v = pl[k]
        if isinstance(v, str):
            print(f"    {k:<22}: \"{v[:80]}{'…' if len(v)>80 else ''}\"")
        elif isinstance(v, list):
            print(f"    {k:<22}: [{len(v)} items] {str(v[:2])[1:-1]}{'…' if len(v)>2 else ''}")
        else:
            print(f"    {k:<22}: {v}")

#  7. FINAL SUMMARY  
print("\n" + "="*70)
print("  [7] FINAL SUMMARY")
print("="*70)
print(f"  Snapshot          : ${SNAPSHOT_NAME}")
print(f"  Collection status : {status}")
print(f"  Total points      : {total_points:,}")
print(f"  Indexed vectors   : {indexed_vecs:,}")
print(f"  Segments          : {segments}")
if papers_complete:
    avg_chunks_true = sum(papers_complete.values()) / len(papers_complete)
    est_papers = int(total_points / avg_chunks_true)
    print(f"  Est. unique papers: {est_papers:,}  (avg {avg_chunks_true:.1f} chunks/paper from payload)")
print(f"  Dense storage est : {dense_gb:.2f} GB")
print(f"  Generated at      : {datetime.datetime.utcnow().isoformat()}Z")

# save JSON report  
report = {
    "generated_at"       : datetime.datetime.utcnow().isoformat() + "Z",
    "snapshot"           : "${SNAPSHOT_NAME}",
    "qdrant_version"     : ver,
    "collection"         : coll,
    "cluster"            : cluster,
    "sample_stats": {
        "points_scrolled"        : fetched,
        "unique_arxiv_ids"       : len(unique_papers),
        "estimated_total_papers" : int(total_points / (fetched / len(unique_papers))) if unique_papers and fetched else None,
        "by_year"                : dict(by_year.most_common()),
        "by_category"            : dict(by_category.most_common(30)),
        "by_chunk_type"          : dict(by_chunk_type.most_common()),
        "by_discipline"          : dict(by_discipline.most_common()),
    },
    "payload_sample"     : scroll1,
}

with open(REPORT, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  Full JSON report saved to: {REPORT}")
PYEOF

echo ""
echo "=========================================="
echo " Health check complete."
echo " Full report: ${REPORT}"
echo " Server log:  ${LOG}"
echo "=========================================="
