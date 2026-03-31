# Section Title Normalization — Post-Index Array Job Plan

## 🎯 Refined Design: Conservative Overwrite with Audit Logs

After indexed corpus completes, run a **parallelized post-index job** to normalize section titles across ~26.4M sections to canonical IMRAD labels. Key principle: **only overwrite when confident**, preserve auditability.

---

## 📊 Target Canonical Labels (Enum)

Exactly 8 labels. All classifier output maps to these:

```python
CANONICAL_SECTIONS = {
    "Abstract",
    "Introduction",
    "Methods",
    "Results",
    "Experiments",
    "Related Work",
    "Discussion",
    "Conclusion",
}
```

**Mapping examples during training:**
- Dataset `background` → `Introduction`
- Dataset `methodology` → `Methods`
- Dataset `findings` → `Results`
- Dataset `experimental results` → `Experiments`
- Dataset `prior work` → `Related Work`
- Dataset `concluding remarks` → `Conclusion`

---

## 🏗️ Decision Flow: 3-Step Conservative Pipeline

```python
for section in sections_in_shard:
    title = section.get("section_title", "")
    text = section.get("text", "")[:500]  # First 300-500 chars
    
    # Step 1: Hard Rules (fast, certain)
    if rule_match := hard_rules(title):
        new_title = rule_match
        source = "rule"
        confidence = 1.0
    else:
        # Step 2: Classifier (title + text context)
        pred, conf = classifier.predict(title, text)
        
        # Step 3: Confidence Gate (don't corrupt on low confidence)
        if conf >= THRESHOLD:  # e.g., 0.85
            new_title = pred
            source = "classifier"
            confidence = conf
        else:
            new_title = title  # LEAVE UNCHANGED if uncertain
            source = "skipped"
            confidence = conf
    
    # Patch all chunks with this section
    update_qdrant_section_title(section_id, new_title)
    
    # Log audit trail
    write_audit_log({
        "paper_id": section.paper_id,
        "section_id": section.id,
        "old_title": title,
        "new_title": new_title,
        "source": source,
        "confidence": confidence,
    })
```

---

## 📋 Step 1: Hard Rules

**Conservative rule set** (only match when very certain):

```python
HARD_RULES = {
    # Exact matches
    "introduction": "Introduction",
    "abstract": "Abstract",
    "related work": "Related Work",
    
    # Common variants
    "methods": "Methods",
    "methodology": "Methods",
    "method": "Methods",
    "experimental setup": "Experiments",
    "experimental design": "Experiments",
    "experiments": "Experiments",
    
    "results": "Results",
    "findings": "Results",
    
    "discussion": "Discussion",
    "discussions": "Discussion",
    "implications": "Discussion",
    
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "concluding remarks": "Conclusion",
    "summary": "Conclusion",
}

def hard_rules(title: str) -> Optional[str]:
    normalized = title.lower().strip()
    # Remove leading numbers (e.g., "3.1 Methods")
    normalized = re.sub(r"^\d+[\.\-]\s*", "", normalized)
    return HARD_RULES.get(normalized)
```

**Design principle:** Only include rules where you're 99%+ confident. Leave edge cases for classifier.

---

## 📋 Step 2: Classifier (Trained Model)

**Input:** `title + [SEP] + first_text`  
**Output:** Canonical label + confidence score

### Training Data

Source: ~500-1000 labeled examples from top UNARXIVE titles:

```
title: "Experimental Results"
text: "We present our findings on the benchmark datasets..."
label: "Experiments"
---
title: "Setup"
text: "The experiment was conducted using the following setup..."
label: "Experiments"
---
title: "Datasets"
text: "We use three public datasets: ImageNet, COCO, and..."
label: "Methods"  # Context signals "Methods" more than "Results"
```

### Model Architecture

Use a lightweight finetuned classifier:

```python
# Option A: DistilBERT (small, fast)
# Option B: Cross-encoder trained on your 8-label problem
# Option C: Prompt-based (LLM few-shot, if budget allows)
```

**Recommendation:** **DistilBERT finetuned on 8-label classification** — good speed/accuracy balance.

### Inference

```python
from transformers import pipeline

class SectionClassifier:
    def __init__(self, model_name="section-classifier-distilbert"):
        self.pipe = pipeline("text-classification", model=model_name)
    
    def predict(self, title: str, text: str, threshold: float = 0.85) -> tuple[str, float]:
        """
        Args:
            title: Section title (normalized)
            text: First 300-500 chars of section text
            threshold: Only return prediction if confidence >= threshold
        
        Returns:
            (label, confidence) or (original_title, low_conf) if below threshold
        """
        input_text = f"{title} [SEP] {text}"
        result = self.pipe(input_text, top_k=None)[0]
        
        label = result["label"]
        confidence = result["score"]
        
        return label, confidence
```

---

## 📋 Step 3: Confidence Gate

**Policy:**
- If hard rule matched → **overwrite** (confidence = 1.0)
- If classifier confidence >= 0.85 → **overwrite** (confidence = conf)
- If classifier confidence < 0.85 → **leave original** (source = "skipped")

**Rationale:** You're editing in place with no backup of original title. Better to leave ambiguous titles unchanged than corrupt them.

---

## 🔧 Processing Unit: Per-Section (Not Per-Chunk)

**Key insight:** Process sections, not chunks.

Why:
- Sections are stable units (titles don't vary within a section)
- Much cheaper: ~26M sections vs ~46M chunks
- Avoids inconsistent labels inside one section

```python
# Group chunks by section within each paper
sections = group_chunks_by_section(chunks)  # {section_id: [chunk1, chunk2, ...]}

for section_id, chunks_in_section in sections.items():
    title = chunks_in_section[0].section_title  # All chunks have same title
    
    # Classify once
    new_title = classify_section(title, chunks_in_section[0].text)
    
    # Patch all chunks
    for chunk in chunks_in_section:
        qdrant_client.update_payload(
            collection_name="arxiv_chunks",
            points_data=[{
                "id": chunk.chunk_uid,
                "payload": {
                    "section_title": new_title,
                }
            }]
        )
```

---

## 📝 Audit Logs (Sidecar JSONLs)

**Problem:** If you overwrite `section_title` in place and hit a bug, debugging is hard.  
**Solution:** Write sidecar audit logs as job artifacts (not in Qdrant).

Per-task output: `_data/normalization_audit/task_{id}.jsonl`

```json
{
  "timestamp": "2026-03-31T10:45:23Z",
  "paper_id": "2310.00826",
  "section_id": "sec_intro_2310.00826",
  "chunk_count": 3,
  "old_title": "Background Material",
  "new_title": "Introduction",
  "source": "classifier",
  "confidence": 0.92,
  "rule_matched": false
}
```

**Aggregation after run:**

```python
def generate_audit_summary():
    all_logs = read_jsonl("_data/normalization_audit/*.jsonl")
    
    summary = {
        "total_sections_processed": len(all_logs),
        "by_source": Counter(log["source"] for log in all_logs),
        "avg_confidence": mean(log["confidence"] for log in all_logs),
        "changed_count": sum(1 for log in all_logs if log["old_title"] != log["new_title"]),
        "skipped_count": sum(1 for log in all_logs if log["source"] == "skipped"),
    }
    
    write_json("_data/normalization_audit/SUMMARY.json", summary)
```

---

## 🚀 Array Job Design

**Script:** `scripts/normalize_sections_array_capella.sh`

```bash
#!/bin/bash
#SBATCH --array=0-63
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30:00

TASK_ID=${SLURM_ARRAY_TASK_ID}
TOTAL_TASKS=${SLURM_ARRAY_TASK_MAX}

python -m src.normalization.normalize_sections_worker \
  --task-id $TASK_ID \
  --total-tasks $TOTAL_TASKS \
  --collection arxiv_chunks_v2 \
  --confidence-threshold 0.85 \
  --batch-size 2000 \
  --audit-dir _data/normalization_audit
```

**Processing:**
- Each task: scroll Qdrant collection
- Filter: `chunk_id % total_tasks == task_id` (even distribution)
- Group by section within each paper
- Classify and update in batches
- Write audit logs

**Scale estimate:**
- ~46M chunks ÷ 64 tasks = ~720k chunks/task
- Batch size 2000 → ~360 API calls/task
- Time: ~15-20 min/task (depending on inference latency)

---

## 📁 Deliverables

| Path | Purpose |
|------|---------|
| `src/normalization/rules.py` | Hard rule definitions + `hard_rules()` function |
| `src/normalization/classifier.py` | `SectionClassifier` class (wraps finetuned DistilBERT) |
| `src/normalization/normalizer.py` | Main orchestrator: rules → classifier → confidence gate |
| `src/normalization/normalize_sections_worker.py` | Worker: scrolls Qdrant, patches in-place, logs audit |
| `src/normalization/audit.py` | Audit log I/O + summary generation |
| `scripts/normalize_sections_array_capella.sh` | SLURM array job launcher |
| `scripts/normalize_sections_local.py` | Dev/test runner (single-threaded) |
| `tests/test_classifier.py` | Unit tests for classifier |
| `tests/test_rules.py` | Unit tests for hard rules |

---

## 🔄 Operational Workflow

### 1. Train Classifier (One-time)

```bash
# Prepare training data (~500-1000 examples)
python scripts/prepare_training_data.py

# Finetune DistilBERT
python scripts/train_section_classifier.py \
  --output-path models/section_classifier_distilbert

# Evaluate
python scripts/eval_section_classifier.py
```

### 2. Test Locally

```bash
# Run on 100 chunks from local Qdrant
python scripts/normalize_sections_local.py \
  --limit 100 \
  --confidence-threshold 0.85
```

### 3. Deploy as Array Job

```bash
# Submit to Capella HPC
sbatch --array=0-63 scripts/normalize_sections_array_capella.sh

# Monitor
watch 'sacct -j <job_id> --format=jobid,state,elapsed,nodelist%20'

# Check logs after completion
tail _data/normalization_audit/SUMMARY.json
```

---

## ⚠️ Important Safeguards

1. **Backup before running:** Snapshot Qdrant collection before normalization
2. **Test on subset first:** Run on 100 chunks locally, inspect audit logs
3. **Monitor confidence distribution:** If avg confidence < 0.80, revisit classifier training
4. **Audit log review:** Check `SUMMARY.json` for skipped/unchanged counts
5. **Validate retrieval:** Test that section filtering still works correctly

---

## ✅ Success Criteria

After normalization completes:

- ✓ All non-skipped sections have `section_title` ∈ canonical labels
- ✓ Skipped sections (low confidence) preserve original title
- ✓ Audit logs capture every change (source, confidence, old→new)
- ✓ Retrieval filter `section_title: "Methods"` returns high-quality results
- ✓ No silent corruptions (confidence gate prevents bad overwrites)
- ✓ `SUMMARY.json` shows breakdown by source (rule / classifier / skipped)

---

## 📚 Next Steps

1. **Finalize hard rules** (confirm conservative list)
2. **Prepare training data** (~500 labeled examples from UNARXIVE top titles)
3. **Finetune DistilBERT** on 8-label problem
4. **Implement local test runner** (script 2 above)
5. **Build array job worker** (script 3 above)
6. **Schedule post-index run** with full corpus

---
