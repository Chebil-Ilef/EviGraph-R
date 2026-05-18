# Checkpoints vs Volumes vs Snapshots

Understanding the difference between these three recovery mechanisms is critical for HPC pipeline reliability.

---

## 1. Checkpoints - Progress Tracking

### What
Metadata files that record "we processed this far"

### Files
```
Phase A (GPU jobs):
/data/horse/ws/<project>/shards/shard_0001.done
/data/horse/ws/<project>/shards/shard_0002.done
...

Phase B (Sequential ingestion):
/data/cat/ws/<project>/ingested_shards.jsonl
```

### Content Example
```json
{"stem": "shard_0001", "status": "INGESTED", "rows": 12500, "timestamp": "2024-01-15T10:30:45Z"}
{"stem": "shard_0002", "status": "INGESTED", "rows": 12500, "timestamp": "2024-01-15T10:31:22Z"}
{"stem": "shard_0003", "status": "INGESTED", "rows": 12500, "timestamp": "2024-01-15T10:32:05Z"}
```

### Purpose
Know which shards to skip on resume

### Code Usage
```python
ingested = _load_ingested_stems()  # Returns: {"shard_0001", "shard_0002", "shard_0003", ...}

for stem in shard_stems:
    if stem in ingested:
        logger.info("Skipping already ingested shard %s", stem)
        continue  # Already processed - don't re-ingest
    
    # Process new shard...
```

### Size Impact
- ~1 KB per line
- 2,800 shards ≈ 140 KB total

---

## 2. Volumes - Live Storage

### What
Host directories mounted into the container so data persists when container restarts

### Setup Configuration
```bash
singularity exec \
  --bind /data/cat/ws/<project>/qdrant_storage:/qdrant/storage \
  --bind /data/cat/ws/<project>/qdrant_snapshots:/qdrant/snapshots \
  qdrant.sif /qdrant
```

### Mapping
```
Inside Container          Host Filesystem
─────────────────         ─────────────────
/qdrant/storage/    ────> /data/cat/ws/<project>/qdrant_storage/
/qdrant/snapshots/  ────> /data/cat/ws/<project>/qdrant_snapshots/
```

### Purpose
Keep data safe when container dies or restarts

### How It Works
```
1. Container running with Qdrant
   ↓
2. Vectors ingested into /qdrant/storage/
   (backed by volume mount to host)
   ↓
3. Job crashes, container dies
   ↓
4. /data/cat/ws/<project>/qdrant_storage/ still exists on host ✅
   (data NOT lost because it was on host filesystem)
   ↓
5. Restart container with same volumes
   ↓
6. Qdrant loads existing data from volume ✅
   (reads from /qdrant/storage/ which still contains all vectors)
```

### Size Impact
- Depends on embedding dimension + vector count
- ~384-dim embeddings × 10M chunks ≈ 15-20 GB
- Continuous I/O as ingestion proceeds

### Key Feature
**Volumes are directory-mounted, NOT stored inside container** - this is the critical difference that makes recovery work.

---

## 3. Snapshots - Database Backups

### What
Point-in-time backup of entire Qdrant collection at checkpoint intervals

### How Created
```python
snapshot_name = create_collection_snapshot(client, "papers")
# Returns: snapshot_2024-01-15T10-30-45Z.snapshot
# Stored in: /data/cat/ws/<project>/qdrant_snapshots/
```

### Metadata Tracked
```json
# qdrant_snapshots/manifest.jsonl
{"snapshot_name": "snapshot_2024-01-15T10-30-45Z", "shard_count": 100, "collection_name": "papers", "created_at": "2024-01-15T10:30:45Z"}
{"snapshot_name": "snapshot_2024-01-15T10-40-22Z", "shard_count": 200, "collection_name": "papers", "created_at": "2024-01-15T10:40:22Z"}
{"snapshot_name": "snapshot_2024-01-15T11-15-33Z", "shard_count": 300, "collection_name": "papers", "created_at": "2024-01-15T11:15:33Z"}
```

### Content
Complete binary copy of all vectors + metadata at that moment

### Creation Schedule
```
After shard 100:   snapshot-1 created ✅
After shard 200:   snapshot-2 created ✅
After shard 300:   snapshot-3 created ✅
...
After all 2,800:   final-snapshot created ✅
```

### Purpose
Restore to known-good state if data corruption occurs

### Size Impact
- Each snapshot ≈ same size as volume
- ~28 snapshots during full run ≈ 420-560 GB total
- Expensive but safety critical

---

## Comparison Table

| Aspect | Checkpoint | Volume | Snapshot |
|--------|-----------|--------|----------|
| **What it tracks** | Which work was done | Live database | Backup copy of DB |
| **Size per unit** | ~1 KB per entry | All vectors | All vectors |
| **Created when** | After each shard | Continuous | Every 100 shards |
| **Survives container death** | ✅ Yes (on disk) | ✅ Yes (mounted) | ✅ Yes (mounted) |
| **Recovery speed** | Fast (just skip) | Fast (already there) | Slow (restore I/O) |
| **Primary use case** | Skip done work | Keep data safe | Restore old state |
| **Recovery overhead** | Zero | Zero | High (I/O bound) |
| **Frequency of need** | Every job restart | Rare (if volume lost) | Rare (DB corruption) |

---

## Real-World Example: Job Dies at Shard 250

### Initial Progress
```
Phase B Ingestion:
  Shard 1-100   ✅ ingested, stored in Qdrant
  Shard 101-200 ✅ ingested, stored in Qdrant
  Shard 201-250 ✅ ingested, stored in Qdrant
  
  → JOB CRASHES ❌
```

### State on Host After Crash
```
/data/cat/ws/<project>/
├── ingested_shards.jsonl          ← CHECKPOINT
│   (knows: shards 1-250 are done)
│
├── qdrant_storage/                ← VOLUME
│   ├── collection_meta/
│   ├── collections/
│   └── ... (all 250 shards' vectors)
│
└── qdrant_snapshots/              ← SNAPSHOTS (backups)
    ├── manifest.jsonl
    │   - snapshot at shard 100: snap-1.snapshot
    │   - snapshot at shard 200: snap-2.snapshot
    │   - snapshot at shard 250: snap-3.snapshot
    └── snap-1.snapshot
    └── snap-2.snapshot
    └── snap-3.snapshot
```

---

## Recovery Scenarios

### Scenario 1: Container Dies (Most Common)

**What happened**
- Singularity container crashed
- Qdrant process died
- But host filesystem intact

**Recovery**
```
$ restart job with resume=True

1. Restart container with same volumes
   ↓
2. Qdrant reads from /qdrant/storage/ ← VOLUME (fast)
   ↓
3. Data intact: all 250 shards present ✅
   ↓
4. Load CHECKPOINT: ingested_shards.jsonl
   ↓
5. Skip shards 1-250 (already done)
   ↓
6. Resume from shard 251 ✅
```

**Result**: Full recovery, minimal time loss

---

### Scenario 2: Volume Deleted (Operator Mistake)

**What happened**
```bash
$ rm -rf /data/cat/ws/<project>/qdrant_storage/  # Oops!
```

**Consequence**
- Live data lost: all 250 shards gone
- VOLUME protection failed
- But SNAPSHOTS exist

**Recovery**
```
1. No volume to mount
   ↓
2. Use latest SNAPSHOT (snap-3.snapshot) ← SNAPSHOTS (slow I/O)
   ↓
3. Restore from snap-3.snapshot
   (restores all 250 shards from backup)
   ↓
4. Load CHECKPOINT: ingested_shards.jsonl
   ↓
5. Skip shards 1-250 (already ingested, now restored)
   ↓
6. Resume from shard 251 ✅
```

**Cost**: 30-60 min for snapshot restore I/O

---

### Scenario 3: Qdrant DB Corruption (Rare)

**What happened**
- Sudden power loss during vector write
- Qdrant DB partially corrupted
- Container runs but data broken
- Queries return garbage

**Recovery**
```
1. Detect corruption in Qdrant (checksums fail)
   ↓
2. Use latest SNAPSHOT (snap-3.snapshot) ← SNAPSHOTS
   ↓
3. Restore clean copy of all 250 shards
   ↓
4. Load CHECKPOINT: ingested_shards.jsonl
   ↓
5. Skip shards 1-250 (verified clean now)
   ↓
6. Resume from shard 251 ✅
```

**Reliability**: Guaranteed clean data from known-good backup

---

## The Safety Stack

```
┌─────────────────────────────────────────────┐
│           CHECKPOINT Layer                  │
│  (ingested_shards.jsonl - "skip markers")   │
│  - Knows which shards are done              │
│  - Very fast to load                        │
│  - Size: ~1 KB per shard                    │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│           VOLUME Layer                      │
│  (/qdrant_storage - live database)          │
│  - Current working copy                     │
│  - Mounted directory survives restarts      │
│  - Size: 15-20 GB for ~10M vectors          │
│  - Try this FIRST on recovery               │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│           SNAPSHOT Layer                    │
│  (snap-*.snapshot - backups)                │
│  - Point-in-time backups                    │
│  - Restore if volume corrupted/lost         │
│  - Size: 28+ copies during full run         │
│  - Try this if volume fails                 │
└─────────────────────────────────────────────┘
```

### Recovery Priority
1. **Try VOLUME first** (fast - already there)
2. **Use CHECKPOINT** (skip re-ingesting)
3. **Restore SNAPSHOT if needed** (slow but reliable)

---

## HPC 7-Day Pipeline

### How All Three Work Together

```
Day 1: Start ingestion
│
├─ Shard 1-100 done
│  └─ CHECKPOINT updated ✅
│  └─ VOLUME has data ✅
│  └─ SNAPSHOT created ✅
│
├─ Shard 101-200 done
│  └─ CHECKPOINT updated ✅
│  └─ VOLUME has data ✅
│  └─ SNAPSHOT created ✅
│
Day 6: At shard 2,700
│  └─ CHECKPOINT: "2,700 shards done"
│  └─ VOLUME: contains all 2,700 shards
│  └─ SNAPSHOTS: 27 backups at each 100-shard interval
│
Day 6, 18:00 → JOB CRASHES ❌
│
Day 6, 18:30 → Restart job with resume=True
│  1. Container restarts with VOLUME mounted ✅
│  2. Load CHECKPOINT: skip shards 1-2,700 ✅
│  3. Resume from shard 2,701 ✅
│
Day 7: Finish remaining 100 shards
│  └─ CHECKPOINT: "2,800 shards done" ✅
│  └─ FINAL SNAPSHOT created ✅
│
Pipeline complete: All 2,800 shards indexed
Data safe: Snapshots backed up to horse → /projects
```

### Without This Stack
```
Day 6, 18:00 → Job crashes
           ↓
No CHECKPOINT → Don't know which shards done
           ↓
No VOLUME → All data lost on container death
           ↓
No SNAPSHOTS → No backup to restore
           ↓
Result: RESTART ENTIRE JOB FROM SHARD 1 ❌❌❌
Time: Lose 5+ days of work
```

---

## Summary

| Layer | Protects Against | Cost | Recovery Time |
|-------|-----------------|------|----------------|
| **CHECKPOINT** | Redundant re-ingestion | 140 KB | None (just skip) |
| **VOLUME** | Container death/restart | 15-20 GB | None (already there) |
| **SNAPSHOT** | Data corruption/loss | 420+ GB | 30-60 min |

**In your system**: All three layers working together = safe, fast, reliable recovery within 7-day HPC window ✅
