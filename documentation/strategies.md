# UnarXive → Qdrant Chunk Schema

### Qdrant Collection: `unarxive_chunks`

Each point in Qdrant represents **one chunk** (abstract or (sub)section).

### 🔹 Chunk Payload Schema (Stored in Qdrant)

```json
{
  
  "chunk_uid": "sha1_hash_here",
  "chunk_type": "abstract|subsection",
  "section_title": "Abstract",
  "embed_text": "Abstract: chunk text here ... ...",

  "spans": {
    "cite_spans": [
      {
        "start": 120,
        "end": 120,
        "work_id": "doi:10.1109/cvpr52688.2022.01552",
        "doi": "10.1109/cvpr52688.2022.01552",
        "openalex_id": "",
        "arxiv_id": "2310.00825",
        "bib_entry_raw": ""
      },
      {
        "start": 200,
        "end": 200,
        "work_id": "unresolved:84f801bbe19df3f203591520e777c0d61492c007",
        "doi": "",
        "openalex_id": "",
        "arxiv_id": "",
        "bib_entry_raw": "Smith, J. (2020). An Unresolved Citation. Journal of Unresolved Research, 10(2), 100-110."
      }
    ]
  },

    "paper_doi": "10.1109/cvpr52688.2022.01553",
    "paper_id_arxiv": "2310.00826",
    "title": "Large Scale Masked Autoencoding for Reducing Label Requirements on SAR Data",
    "authors": ["Matt Allen", "Francisco Dorr"],
    "categories": ["cs.CV", "eess.IV"],
    "year": 2023,
    "cited_by_count": 5,
    "language": null,
    "discipline": null
}
```


# Chunking Strategy

### Abstracts

→ default 1 chunk, do not split.

- **Default:** keep abstract as **one chunk**
- **If abstract > ~300 tokens:** split into chunks with overlap

Why: abstracts are already dense; splitting short ones hurts more than helps.

Gets its own chunk_type = "abstract" so retrieval can weight or filter it separately.

### Subsections

Sections → 1 chunk if ≤ 400 tokens; sliding window otherwise.

Section titles are your semantic boundaries : already better than any chunker can do.

Only ~10% of sections (long intros/related work) exceed 400 tokens.
For those: sliding window of *350 tokens, 50-token overlap*, split at sentence boundaries.
⇒ determinitic and fast

No fixed-size chunking, no semantic chunker (LLM-based splitting) because  undeterministic and slow.

**Why NOT a semantic chunker?**

- Section keys in unarXive are already human-defined semantic units.
- Semantic chunkers add latency, cost, and inconsistency for marginal gain when structure is already present.

**Cite spans**

Resolve ref_id hash → DOI at parse time using bib_entries.
Store resolved DOIs inside each chunk's cite_spans list.
This way retrieval consumers never need the raw bib dict.

# Embedding strategy

### Embed **one vector per chunk**

- abstract chunk(s)
- section/subsection chunks

### Query/passage formatting

- For `intfloat/e5-base-v2`:
    - chunk embed: `"passage: {text}"`
    - query embed: `"query: {q}"`
- For others (`jina nano / embeddinggemma / Qwen3-Embedding`): start with plain text; we’ll keep this in a wrapper so switching is config-only.

### When to normalize vectors

- If you use cosine similarity in Qdrant: normalize embeddings (unit length) consistently for both docs and queries.
- We’ll decide this in Task 3 when we pick exact Qdrant distance setting (Cosine vs Dot).

# Fast indexing in Qdrant (what matters for speed)

- One collection `unarxive_chunks`
- ANN index: **HNSW**
- Store payload fields needed for filtering:
    - `paper_id_arxiv`, `chunk_type`, `section_title`, `paper.year`, `paper.categories`
- Use **batch upserts** (your N=6 batches) and prefer larger payload write chunks for throughput.

# Hybrid retrieval in Qdrant

### Option 1 (recommended MVP): Dense + BM25 text search + RRF

This is the closest to what SQuAI did, but now inside Qdrant.

1. **Dense search**
- search using the chunk embedding vector
1. **BM25 / text search**
- search on payload field `text` using Qdrant text search (BM25)
1. **Fuse results**
- Reciprocal Rank Fusion (RRF)
    
    Qdrant’s own learning materials and docs show hybrid pipelines using dense+sparse with RRF.
    

✅ Works with *all* your dense models (E5, Jina, Gemma, Qwen3)

✅ Very strong for scientific terms, acronyms, dataset names

✅ Fast and simple

### Option 2: Dense + Sparse vectors in Qdrant (best with BGE-M3)

If you use **BGE-M3**, you can store:

- dense vector
- sparse vector (token weights)

Then hybrid search is:

- query dense vector + query sparse vector
- combine (again with RRF or weighted sum)

BGE-M3 explicitly supports sparse retrieval alongside dense.

# Optimizations for scale

[https://qdrant.tech/articles/indexing-optimization/](https://qdrant.tech/articles/indexing-optimization/?utm_source=chatgpt.com) 

### OPTION 1 — Dense + BM25

| Component | 💻 Laptop (Prototype) | 🖥 HPC (Full Scale 2.8M papers) | Why |
| --- | --- | --- | --- |
| Embedding models | E5 / Jina / Gemma / Qwen | Same | Dense-only models |
| Hybrid method | Dense + BM25 + RRF | Same | Matches “E5 + BM25” baseline |
| Distance metric | Cosine | Cosine | Standard for normalized embeddings |
| HNSW `m` | 16 | 32 | Higher = better recall but more RAM |
| HNSW `ef_construct` | 64 | 128 | Better index quality at scale |
| Query-time `ef` | 64 | 64–128 | Recall/latency tuning knob |
| Vectors storage | In RAM | `on_disk = true` (mmap) | Reduce RAM usage |
| Quantization | ❌ None | ✅ Scalar quantization | ~4× memory reduction |
| Payload storage | In RAM | OnDisk payload | Chunk text is large |
| Full-text index | ✅ on `text` | ✅ on `text` | Enables BM25 |
| Payload indexes | Basic filters | Basic filters | Faster filtered queries |
| Complexity | Low | Medium | Production ready |

### OPTION 2 — Dense + Sparse (BGE-M3 Hybrid)

| Component | 💻 Laptop (Prototype) | 🖥 HPC (Full Scale) | Why |
| --- | --- | --- | --- |
| Model | BGE-M3 | BGE-M3 | Produces dense + sparse |
| Vectors stored | Dense + Sparse | Dense + Sparse | True hybrid retrieval |
| Hybrid method | Dense + Sparse + RRF | Same | No separate BM25 needed |
| HNSW `m` | 16 | 32 | Dense graph tuning |
| HNSW `ef_construct` | 64 | 128 | Better recall at scale |
| Query-time `ef` | 64 | 64–128 | Tune recall/latency |
| Vectors storage | RAM | `on_disk = true` | Scale safely |
| Quantization | ❌ None | ✅ Scalar (dense only) | Save RAM |
| Sparse index | Default | Default | Already inverted index |
| Payload storage | RAM | OnDisk | Chunk text large |
| Complexity | Medium | Higher | More engineering |