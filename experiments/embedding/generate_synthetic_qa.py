import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_key_phrases(text: str, max_phrases: int = 2) -> list[str]:

    sentences = [s.strip() for s in text.split(".")[:max_phrases] if s.strip()]
    return [s for s in sentences if len(s) > 10]


def generate_question_from_phrase(phrase: str, paper_title: Optional[str] = None) -> str:

    phrase = re.sub(r"\([^)]*\)", "", phrase)
    phrase = re.sub(r"\[[^\]]*\]", "", phrase)
    phrase = phrase.strip()
    
    if not phrase:
        return ""
    
    words = phrase.split()[:10]
    
    questions = [
        f"What is {phrase[:80]}?",
        f"How does {phrase[:80]}?",
        f"Explain {phrase[:80]}.",
        f"Describe {phrase[:80]}.",
    ]
    
    return questions[0] 


def load_chunks(chunks_path: Path, sample_size: Optional[int] = None) -> list[dict]:

    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                if sample_size and len(chunks) >= sample_size:
                    break
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)
    return chunks


def generate_qa_records(chunks: list[dict]) -> list[dict]:

    qa_records = []
    question_id_counter = 0
    
    for chunk_idx, chunk in enumerate(chunks):
        paper_id = chunk.get("paper_id", f"paper_{chunk_idx}")
        chunk_uid = chunk.get("uid", chunk.get("chunk_uid", f"chunk_{chunk_idx}"))
        section_title = chunk.get("section_title", "Untitled Section")
        embed_text = chunk.get("embed_text", "")
        
        if not embed_text or len(embed_text) < 20:
            continue  # Skip very short chunks
        
        # Extract key phrases
        phrases = extract_key_phrases(embed_text, max_phrases=2)
        
        for phrase in phrases:
            if not phrase or len(phrase) < 15:
                continue
            
            question_id = f"q_{question_id_counter:06d}"
            question_id_counter += 1
            
            query = generate_question_from_phrase(phrase)
            
            if not query:
                continue
            
            answer_strings = []
            answer_strings.append(phrase[:100])
            
            first_sentence = embed_text.split(".")[0].strip()
            if first_sentence and first_sentence != phrase[:len(first_sentence)]:
                answer_strings.append(first_sentence[:100])
            
            qa_record = {
                "question_id": question_id,
                "query": query,
                "gold_paper_id": paper_id,
                "gold_chunk_uid": chunk_uid,
                "gold_section_title": section_title,
                "gold_answer_strings": answer_strings,
            }
            
            qa_records.append(qa_record)
    
    logger.info("Generated %d Q&A records from %d chunks", len(qa_records), len(chunks))
    return qa_records


def save_qa_records(qa_records: list[dict], output_path: Path) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in qa_records:
            f.write(json.dumps(record) + "\n")
    logger.info("Saved %d Q&A records to %s", len(qa_records), output_path)


def main():
    p = argparse.ArgumentParser(
        description="Generate synthetic Q&A dataset with gold ground truth from chunks."
    )
    p.add_argument(
        "--chunks", required=True, type=Path,
        help="Path to chunks JSONL file.",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Path to output synthetic_qa.jsonl.",
    )
    p.add_argument(
        "--sample-size", type=int, default=None,
        help="Limit to first N chunks (for testing).",
    )
    args = p.parse_args()
    
    chunks = load_chunks(args.chunks, sample_size=args.sample_size)
    qa_records = generate_qa_records(chunks)
    save_qa_records(qa_records, args.output)
    
    logger.info("✓ Done!")


if __name__ == "__main__":
    main()
