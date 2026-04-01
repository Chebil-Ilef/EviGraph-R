import argparse
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ABBR2ID = {
    "i": 0,  # Introduction
    "m": 1,  # Methods
    "r": 2,  # Results
    "w": 3,  # Related Work
    "d": 4,  # Discussion
}

ID2LABEL = {
    0: "Introduction",
    1: "Methods",
    2: "Results",
    3: "Related Work",
    4: "Discussion",
}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def load_imrad_dataset(dataset_id: str = "saier/unarXive_imrad_clf") -> DatasetDict:

    logger.info(f"Loading dataset: {dataset_id}")
    
    try:
        dataset = load_dataset(dataset_id)
        logger.info(f"Dataset loaded. Splits: {list(dataset.keys())}")
        logger.info(f"Train split size: {len(dataset['train'])}")
        
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def preprocess_function(examples, tokenizer, max_length: int = 512):

    texts = examples["text"]
    
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None,
    )
    
    label_ids = [ABBR2ID.get(label, 0) for label in examples["label"]]
    
    encodings["labels"] = label_ids
    return encodings


def compute_metrics(eval_pred):

    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


class WeightedTrainer(Trainer):
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Use weighted cross entropy if class weights provided
        if self.class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(self.class_weights, device=logits.device, dtype=torch.float)
            )
            loss = loss_fn(logits, labels)
        else:
            loss = outputs["loss"] if "loss" in outputs else None
            if loss is None:
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(
        description="Train IMRAD section classifier on HuggingFace Hub dataset"
    )
    parser.add_argument(
        "--dataset-id",
        default="saier/unarXive_imrad_clf",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--model-id",
        default="distilbert-base-uncased",
        help="Base model from HuggingFace Hub",
    )
    parser.add_argument(
        "--output-dir",
        default="./models/section_classifier",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=32,
        help="Batch size per device",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push trained model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-repo-id",
        default=None,
        help="Hub repo ID for push (required if --push-to-hub)",
    )
    parser.add_argument(
        "--eval-strategy",
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Evaluation interval (if eval_strategy=steps)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Validate
    if args.push_to_hub and not args.hub_repo_id:
        parser.error("--hub-repo-id required when using --push-to-hub")
    
    logger.info(f"Args: {args}")
    
    dataset = load_imrad_dataset(args.dataset_id)
    
    # Calculate class weights for imbalanced dataset
    logger.info("Calculating class weights...")
    label_counts = Counter(dataset["train"]["label"])
    total_samples = len(dataset["train"])
    
    class_weights = np.array([
        total_samples / (len(label_counts) * label_counts.get(abbr, 1))
        for abbr in ["i", "m", "r", "w", "d"]
    ])
    class_weights = class_weights / class_weights.sum() * len(class_weights)  # normalize
    
    logger.info("Label distribution:")
    for abbr, count in sorted(label_counts.items()):
        label_id = ABBR2ID[abbr]
        label_name = ID2LABEL[label_id]
        weight = class_weights[label_id]
        pct = 100 * count / total_samples
        logger.info(f"  {abbr} ({label_name}): {count:,} ({pct:.1f}%) - weight: {weight:.2f}")
    
    logger.info(f"Loading base model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    
    logger.info("Preprocessing dataset...")
    
    def preprocess_fn(examples):
        return preprocess_function(examples, tokenizer, args.max_seq_length)
    
    dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=["_id", "text", "label"],  # Remove raw columns after tokenization
        desc="Tokenizing dataset",
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo_id,
        hub_strategy="every_save",
        hub_private_repo=False,  # Make repo public
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        seed=args.seed,
        dataloader_pin_memory=True if args.device == "cuda" else False,
        dataloader_num_workers=4,
        remove_unused_columns=True,
    )
    
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else dataset["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],  # Increased patience
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    logger.info(f"Training completed. Results: {train_result}")
    
    # Evaluate on test set
    if "test" in dataset:
        logger.info("Evaluating on test set...")
        eval_result = trainer.evaluate(eval_dataset=dataset["test"])
        logger.info(f"Test results: {eval_result}")
    
    # Save model locally
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Push to Hub if requested
    if args.push_to_hub:
        logger.info(f"Pushing model to Hub: {args.hub_repo_id}")
        trainer.push_to_hub()
        logger.info("✓ Model pushed to Hub successfully")
    
    logger.info("Training complete!")
    
    return {
        "model_dir": str(output_dir),
        "hub_repo_id": args.hub_repo_id if args.push_to_hub else None,
        "metrics": train_result.metrics if hasattr(train_result, "metrics") else None,
    }


if __name__ == "__main__":
    main()
