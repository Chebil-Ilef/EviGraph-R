import argparse
import logging
import os
from pathlib import Path
import numpy as np
import torch
import json as _json
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from datetime import datetime
from huggingface_hub import HfApi

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

LABEL2ID = {"background": 0, "method": 1, "result": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--output_dir", default="./scicite_output")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", default=True,
                   help="Mixed precision — disable with --no-fp16 on CPU-only nodes")
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    # HPC: use local scratch for cache to avoid quota issues on $HOME
    p.add_argument("--cache_dir", default=os.environ.get("HPC_SCRATCH", None),
                   help="HuggingFace cache dir — defaults to $HPC_SCRATCH if set")
    p.add_argument("--push_to_hub", action="store_true", default=False,
                   help="Push the fine-tuned model to the HuggingFace Hub after training")
    p.add_argument("--hub_model_id", default=None,
                   help="Hub repo id to push to, e.g. 'myuser/scibert-finetuned-scicite'. "
                        "Defaults to '<base-model-name>-finetuned-scicite'.")
    p.add_argument("--hub_token", default=os.environ.get("HF_TOKEN", None),
                   help="HuggingFace write token (falls back to $HF_TOKEN env var)")
    return p.parse_args()


def _load_jsonl(path: str):

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = _json.loads(line)
            rows.append({
                "string": str(d.get("string") or ""),
                "label":  str(d.get("label") or ""),
            })
    return rows


def load_scicite(cache_dir):

    import tarfile, urllib.request, tempfile, os

    log.info("Loading SciCite dataset …")
    url = "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz"

    # Use cache_dir so the tar isn't re-downloaded on re-runs
    cache_root = cache_dir or tempfile.gettempdir()
    tar_path   = os.path.join(cache_root, "scicite.tar.gz")
    data_dir   = os.path.join(cache_root, "scicite_extracted")

    if not os.path.isdir(data_dir):
        if not os.path.isfile(tar_path):
            log.info(f"Downloading {url} …")
            urllib.request.urlretrieve(url, tar_path)
        log.info("Extracting …")
        with tarfile.open(tar_path, "r:gz") as t:
            t.extractall(data_dir)

    base = os.path.join(data_dir, "scicite")
    ds = DatasetDict({
        "train":      Dataset.from_list(_load_jsonl(os.path.join(base, "train.jsonl"))),
        "validation": Dataset.from_list(_load_jsonl(os.path.join(base, "dev.jsonl"))),
        "test":       Dataset.from_list(_load_jsonl(os.path.join(base, "test.jsonl"))),
    })
    log.info(f"Train: {len(ds['train'])}  Val: {len(ds['validation'])}  Test: {len(ds['test'])}")
    return ds


def tokenize_dataset(ds, tokenizer, max_length):
    def tokenize(batch):
        enc = tokenizer(
            batch["string"],          # citation context sentence
            truncation=True,
            max_length=max_length,
        )
        enc["labels"] = [LABEL2ID[l] for l in batch["label"]]
        return enc

    log.info("Tokenizing …")
    ds = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
    ds.set_format("torch")
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    return {"macro_f1": macro_f1, "micro_f1": micro_f1}


def compute_class_weights(ds_train):
    labels = [LABEL2ID[l] for l in ds_train["label"]]
    weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    log.info(f"Class weights: background={weights[0]:.3f}, method={weights[1]:.3f}, result={weights[2]:.3f}")
    return torch.tensor(weights, dtype=torch.float)


def full_report(trainer, ds, split="test"):
    preds_out = trainer.predict(ds[split])
    preds = np.argmax(preds_out.predictions, axis=-1)
    labels = preds_out.label_ids
    print("\n" + "=" * 60)
    print(f"Classification report on [{split}] split")
    print("=" * 60)
    print(classification_report(labels, preds, target_names=list(LABEL2ID.keys())))
    
    # Return metrics for model card
    report = classification_report(labels, preds, target_names=list(LABEL2ID.keys()), output_dict=True)
    accuracy = (preds == labels).mean()
    return {
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "report": report,
        "preds": preds,
        "labels": labels
    }


def generate_model_card(args, trainer, test_metrics, training_history, best_epoch):
    
    report = test_metrics["report"]
    
    # Build per-class rows
    class_rows = []
    for label_name in LABEL2ID.keys():
        if label_name in report:
            r = report[label_name]
            class_rows.append(
                f"| {label_name} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1-score']:.2f} | {int(r['support'])} |"
            )
    class_table = "\n".join(class_rows)
    
    # Build training history table with proper train loss aggregation
    history_rows = []
    epoch_train_losses = {}  # epoch_num -> list of train losses
    
    # First pass: collect training losses from all log entries
    for entry in trainer.state.log_history:
        if "loss" in entry and "epoch" in entry:
            epoch_num = int(entry["epoch"])
            if epoch_num not in epoch_train_losses:
                epoch_train_losses[epoch_num] = []
            epoch_train_losses[epoch_num].append(entry["loss"])
    
    # Second pass: build table from eval entries paired with aggregated train loss
    for metrics, epoch_num in training_history:
        if "eval_macro_f1" in metrics:  # Only eval epochs
            epoch_int = int(epoch_num)
            # Get average training loss for this epoch
            train_loss = "N/A"
            if epoch_int in epoch_train_losses and epoch_train_losses[epoch_int]:
                train_loss = f"{sum(epoch_train_losses[epoch_int]) / len(epoch_train_losses[epoch_int]):.4f}"
            
            eval_loss = f"{metrics.get('eval_loss', 0):.4f}" if "eval_loss" in metrics else "N/A"
            macro_f1 = f"{metrics.get('eval_macro_f1', 0):.4f}" if "eval_macro_f1" in metrics else "N/A"
            micro_f1 = f"{metrics.get('eval_micro_f1', 0):.4f}" if "eval_micro_f1" in metrics else "N/A"
            
            history_rows.append(f"  {epoch_int} | {train_loss:<10} | {eval_loss:<10} | {macro_f1:<8} | {micro_f1:<8}")
    
    history_table = "\n".join(history_rows) if history_rows else "  (metrics not available)"
    
    model_card = f"""---
language: en
license: apache-2.0
tags:
  - scicite
  - citation-intent
  - scientific-text
  - sequence-classification
model-index:
- name: scibert-scicite-classifier
  results:
  - task:
      name: text-classification
      type: sequence-classification
    dataset:
      name: scicite
      type: scicite
    metrics:
    - name: Accuracy
      type: accuracy
      value: {test_metrics['accuracy']:.4f}
    - name: Macro F1
      type: f1
      value: {test_metrics['macro_f1']:.4f}
---

# SciBERT Fine-tuned for SciCite Intent Classification

This model is a fine-tuned version of [allenai/scibert_scivocab_uncased](https://huggingface.co/allenai/scibert_scivocab_uncased) on the [SciCite dataset](https://github.com/allenai/scicite).

## Model Description

**Base Model:** AllenAI's SciBERT pre-trained on scientific text with a domain-specific vocabulary.

**Task:** Citation Intent Classification — predicting why an author is citing another work (background, method, or result).

**Labels:** 
- `background` (0): Citations providing prior work context
- `method` (1): Citations of techniques or methodologies
- `result` (2): Citations comparing or contrasting experimental results

## Results

Achieved on the SciCite test set:

| Metric | Score |
|--------|-------|
| Accuracy | {test_metrics['accuracy']:.2%} |
| Macro F1 | {test_metrics['macro_f1']:.4f} |
| Weighted F1 | {test_metrics['weighted_f1']:.4f} |

**Per-class performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
{class_table}

## Intended Uses & Limitations

**Intended Use:** Automatically classify citation intents in academic papers to improve literature mining, knowledge graph construction, and semantic search applications.

**Limitations:** Model trained on arXiv scientific abstracts; may not generalize to other domains (biomedical, legal, etc.). Best performance on background/method classes; result class has lower precision due to class imbalance.

## Training and Evaluation Data

**Dataset:** [SciCite](https://github.com/allenai/scicite) — 8,243 training examples, 916 validation, 1,861 test (citation contexts from arXiv papers).

**Format:** Citation sentence + class label. Max length: {args.max_length} tokens. Split: 80% train, 10% val, 10% test.

## How to Use

### Installation

```bash
pip install transformers torch
```

### Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("lostelf/scibert_scicite_finetuned")
model = AutoModelForSequenceClassification.from_pretrained("lostelf/scibert_scicite_finetuned")

text = "We use the BERT architecture as in Devlin et al."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length={args.max_length})

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()

labels = {{0: "background", 1: "method", 2: "result"}}
print(f"Predicted: {{labels[predicted_class]}}")
```

### Batch Prediction

```python
texts = [
    "We build on the transformer framework introduced by Vaswani et al.",
    "Our implementation follows the optimization procedure in Kingma & Ba.",
    "These results exceed prior work by Devlin et al. (BERT)."
]

inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length={args.max_length})
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)
print([labels[p] for p in predictions])
```

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | {args.model_name} |
| Epochs | {args.num_epochs} |
| Batch Size | {args.batch_size} |
| Learning Rate | {args.lr} |
| Warmup Steps | {int(args.warmup_ratio * 8243 / args.batch_size)} (~{args.warmup_ratio*100:.0f}% of training) |
| Weight Decay | {args.weight_decay} |
| Optimizer | AdamW |
| LR Scheduler | linear |
| Gradient Accumulation | 2 steps |
| FP16 | {"Enabled" if args.fp16 else "Disabled"} |

## Training Results

Best checkpoint: Epoch {best_epoch} (macro F1 = {test_metrics['macro_f1']:.4f}). Early stopping patience = 4 epochs.

Training curve (eval epochs):

```
Epoch | Train Loss | Val Loss | Macro F1 | Micro F1
------|------------|----------|----------|----------
{history_table}
```

## Framework Versions

- Python: 3.11
- PyTorch: 2.0+
- Transformers: 4.38+
- Datasets: 2.14+
- Scikit-learn: 1.3+

---

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Training GPU:** {os.environ.get('SLURM_JOB_GPUS', '1x GPU (Capella HPC)')}  
"""
    
    return model_card


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Tokenizer & model
    log.info(f"Loading {args.model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        cache_dir=args.cache_dir,
    )

    # 2. Data
    ds_raw = load_scicite(args.cache_dir)
    ds = tokenize_dataset(ds_raw, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer)
    
    # Log class distribution for diagnostics (helps understand imbalance)
    compute_class_weights(ds_raw["train"])

    # 3. Training args
    # Capella uses SLURM — local_rank is set automatically via torchrun/srun
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=int(args.warmup_ratio * 8243 / args.batch_size),  # converted from ratio
        lr_scheduler_type="linear",
        fp16=args.fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",           # swap to "wandb" or "tensorboard" if available
        seed=args.seed,
        dataloader_num_workers=4, 
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=2,  # Increase effective batch size for more stable gradients
    )

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],  # More patience for sustained improvements
    )

    # 5. Train
    log.info("Starting training …")
    trainer.train()

    # 6. Save best model
    best_dir = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    log.info(f"Best model saved to {best_dir}")

    # 7. Evaluate on test
    test_metrics = full_report(trainer, ds, split="test")
    
    # 8. Generate dynamic model card with actual results
    # Extract training history: collect all eval steps
    training_history = [
        (entry, entry.get("epoch", 0))
        for entry in trainer.state.log_history
        if "eval_loss" in entry  # Only eval steps
    ]
    best_epoch = getattr(trainer.state, "best_epoch", args.num_epochs)
    
    model_card_content = generate_model_card(args, trainer, test_metrics, training_history, best_epoch)
    
    # Save to best_model directory as README.md
    readme_path = os.path.join(best_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card_content)
    log.info(f"Dynamic model card generated and saved to {readme_path}")

    # 9. Push to HuggingFace Hub
    if args.push_to_hub:
        from huggingface_hub import upload_file
        
        if not args.hub_token:
            raise ValueError(
                "HF write token required for --push_to_hub. "
                "Pass --hub_token or set $HF_TOKEN."
            )
        
        base_name = args.model_name.split("/")[-1]
        if args.hub_model_id:
            hub_model_id = args.hub_model_id
        else:
            # Try to get username from env, fallback to lostelf
            username = os.environ.get("INDEXING_HF_USERNAME", "lostelf")
            hub_model_id = f"{username}/{base_name}_scicite_finetuned"
        
        log.info(f"Pushing model to HuggingFace Hub as '{hub_model_id}' …")
        
        # Push model and tokenizer via trainer
        trainer.push_to_hub(hub_model_id, token=args.hub_token)
        
        # Explicitly upload the dynamic README (trainer.push_to_hub doesn't include it)
        readme_path = os.path.join(best_dir, "README.md")
        if os.path.exists(readme_path):
            log.info("Uploading dynamic README.md …")
            upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=hub_model_id,
                token=args.hub_token,
            )
        
        log.info(f"✓ Model pushed to https://huggingface.co/{hub_model_id}")


if __name__ == "__main__":
    main()