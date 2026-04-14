import argparse
import logging
import os
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
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


def load_scicite(cache_dir):
    log.info("Loading SciCite dataset …")
    ds = load_dataset("allenai/scicite", cache_dir=cache_dir, trust_remote_code=True)
    # SciCite label column is 'label' with string values
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


def full_report(trainer, ds, split="test"):
    preds_out = trainer.predict(ds[split])
    preds = np.argmax(preds_out.predictions, axis=-1)
    labels = preds_out.label_ids
    print("\n" + "=" * 60)
    print(f"Classification report on [{split}] split")
    print("=" * 60)
    print(classification_report(labels, preds, target_names=list(LABEL2ID.keys())))


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

    # 3. Training args
    # Capella uses SLURM — local_rank is set automatically via torchrun/srun
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        fp16=args.fp16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        report_to="none",           # swap to "wandb" or "tensorboard" if available
        seed=args.seed,
        dataloader_num_workers=4, 
        ddp_find_unused_parameters=False,
    )

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
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
    full_report(trainer, ds, split="test")

    # 8. Push to HuggingFace Hub
    if args.push_to_hub:
        base_name = args.model_name.split("/")[-1]
        hub_model_id = args.hub_model_id or f"{base_name}_scicite_finetuned"
        log.info(f"Pushing model to HuggingFace Hub as '{hub_model_id}' …")
        if not args.hub_token:
            raise ValueError(
                "HF write token required for --push_to_hub. "
                "Pass --hub_token or set $HF_TOKEN."
            )
        model.push_to_hub(hub_model_id, token=args.hub_token)
        tokenizer.push_to_hub(hub_model_id, token=args.hub_token)
        log.info(f"Model pushed to https://huggingface.co/{hub_model_id}")


if __name__ == "__main__":
    main()