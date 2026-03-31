import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dataset():

    logger.info("=" * 40)
    logger.info("Checking dataset availability...")
    logger.info("=" * 40)
    
    try:
        from datasets import load_dataset
        
        dataset_id = "saier/unarXive_imrad_clf"
        logger.info(f"Loading dataset: {dataset_id}")
        
        # Load with limited size first
        dataset = load_dataset(
            dataset_id,
            split="train",
            streaming=False,
        )
        
        logger.info(f"✓ Dataset loaded successfully")
        logger.info(f"  - Train split size: {len(dataset)}")
        
        # Check actual labels
        unique_labels = set(dataset['label'])
        logger.info(f"\n✓ Unique labels in dataset: {sorted(unique_labels)}")
        logger.info(f"  - Label mappings:")
        
        label_map = {
            "i": "Introduction",
            "m": "Methods",
            "r": "Results",
            "w": "Related Work",
            "d": "Discussion",
        }
        
        from collections import Counter
        label_counts = Counter(dataset['label'])
        for abbr in sorted(label_counts.keys()):
            full_name = label_map.get(abbr, "Unknown")
            count = label_counts[abbr]
            logger.info(f"    - {abbr} ({full_name}): {count:,}")
        
        # Show sample
        sample = dataset[0]
        logger.info(f"\nSample entry:")
        logger.info(f"  - _id: {sample.get('_id', 'N/A')}")
        logger.info(f"  - label: {sample.get('label', 'N/A')} ({label_map.get(sample.get('label'), 'Unknown')})")
        logger.info(f"  - text: {sample.get('text', 'N/A')[:100]}...")
        logger.info(f"  - text length: {len(sample.get('text', ''))}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Dataset check failed: {e}")
        return False


def check_dependencies():

    logger.info("\n" + "=" * 40)
    logger.info("Checking dependencies...")
    logger.info("=" * 40)
    
    required = [
        "transformers",
        "datasets",
        "torch",
        "sklearn",
    ]
    
    all_ok = True
    for pkg in required:
        try:
            __import__(pkg)
            logger.info(f"✓ {pkg}")
        except ImportError:
            logger.error(f"✗ {pkg} not installed")
            all_ok = False
    
    return all_ok


def check_models():

    logger.info("\n" + "=" * 40)
    logger.info("Checking base model availability...")
    logger.info("=" * 40)
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_id = "distilbert-base-uncased"
        logger.info(f"Loading: {model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=5,  # 5 labels in dataset: i, m, r, w, d
        )
        
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  - Vocab size: {len(tokenizer)}")
        logger.info(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Model check failed: {e}")
        return False


def check_preprocessing():

    logger.info("\n" + "=" * 40)
    logger.info("Testing preprocessing...")
    logger.info("=" * 40)
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Test input
        title = "Experiments"
        text = "We evaluated our method on three benchmark datasets..."
        input_text = f"{title} [SEP] {text[:500]}"
        
        # Tokenize
        tokens = tokenizer(
            input_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        logger.info(f"✓ Preprocessing successful")
        logger.info(f"  - Input: '{input_text[:60]}...'")
        logger.info(f"  - Tokens: {tokens['input_ids'].shape}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Preprocessing check failed: {e}")
        return False


def main():

    logger.info("\n" + "=" * 40)
    logger.info("IMRAD Section Classifier — Setup Verification")
    logger.info("=" * 40 + "\n")
    
    checks = {
        "Dependencies": check_dependencies,
        "Models": check_models,
        "Preprocessing": check_preprocessing,
        "Dataset": check_dataset,
    }
    
    results = {}
    for name, check_fn in checks.items():
        try:
            results[name] = check_fn()
        except Exception as e:
            logger.error(f"Unexpected error in {name}: {e}")
            results[name] = False
    
    logger.info("\n" + "=" * 40)
    logger.info("Summary")
    logger.info("=" * 40)
    
    for name, status in results.items():
        symbol = "✓" if status else "✗"
        logger.info(f"{symbol} {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n✓ All checks passed! Ready to train.")
        return 0
    else:
        logger.error("\n✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
