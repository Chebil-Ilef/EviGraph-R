# unarXive 2024 — Data Quality Report

> **Analysed:** 2,214,129 papers (requested 2,338,911)  
> **Date:** 2026-03-22 00:56:22  
> **Scale target:** 2,338,911 papers  
> **Chunk config:** window=650 tok · overlap=50 tok · no-split≤700 tok  
> **Embedding:** e5-base-v2 · dim=768 · float32

---
##  0 · Dataset Schema

Each paper is a JSONL row. The relevant top-level keys are:

| Key | Type | Notes |
|-----|------|-------|
| `paper_id` | string | arXiv ID, e.g. `2310.00826` — primary identifier |
| `metadata` | dict | Title, authors, categories, DOI — **no `year` field exists** |
| `abstract` | dict | `{text, cite_spans, ref_spans}` |
| `sections` | dict | `{title → {text, cite_spans, ref_spans}}` — **all body content lives here** |
| `bib_entries` | dict | `{sha_hex → {bib_entry_raw, ids{doi, arxiv_id, open_alex_id}}}` |
| `ref_entries` | dict | Figure/table captions, keyed by UUID |

> **Body text:** There is no separate `body_text` field. All body word counts and chunk estimates
> in this report are computed by summing across all entries in the `sections` dict.
>
> **Year:** Not present in `metadata`. Fully derivable from the arXiv ID YYMM prefix:
> `2310.00826` → year **2023**, month **10**. The preprocessor must derive and store this.

---
##  1 · Paper Metadata

| Field | Count | % of papers |
|-------|------:|------------:|
| Title | **2,213,493** | 100.0% |
| Authors | **2,213,493** | 100.0% |
| Year (metadata field) | **0** | 0.0% |
| Year (derived from arXiv ID YYMM) | **1,996,274** | 90.2% |
| Paper DOI | **1,013,429** | 45.8% |
| Paper arXiv ID | **1,996,274** | 90.2% |
| Categories | **2,213,493** | 100.0% |

> ⚠️ **Year is 0% in metadata** but 90.2% derivable from arXiv ID.
> Add `_derive_year(paper_id)` to the preprocessor and store result in the chunk payload.

---
##  2 · Abstract

| | Count | % |
|--|------:|--:|
| Non-empty abstract | **2,214,129** | 100.0% |
| Empty / missing | **0** | 0.0% |

---
##  3 · Body Text

> **What is body word count?** Sum of word counts across all `sections` entries for a paper.
> There is no separate body field — this is computed at analysis time.
>
> **What are p10, p25, p50, p75, p90?** These are *percentiles* estimated from a memory-safe sample.

| | Count | % |
|--|------:|--:|
| Has body text | **2,196,969** | 99.2% |
| Empty / missing | **17,160** | 0.8% |

| Statistic | Words |
|-----------|------:|
| Mean | **5,973** |
| p10 | 1,699 |
| p25 | 2,748 |
| p50 (median) | **4,496** |
| p75 | 7,188 |
| p90 | 10,998 |

---
##  4 · Section Structure

> **What are section titles?** Each key of the `sections` dict is a section title as it
> appeared in the paper — e.g. `"Introduction"`, `"The Obstacle Problem"`, `"3.1 Setup"`.
> The pipeline stores these in `chunk_section` for filtered retrieval.
>
> **IMRAD titles** match standard scientific paper structure keywords (Introduction,
> Method, Results, Discussion, etc.). These are the reliable section-level filters.
>
> **Noise title** = a section title that is purely numeric, roman numeral, or punctuation.

| Metric | Value |
|--------|------:|
| Total sections | **26,384,995** |
| Avg sections / paper | **12.0** |
| Sections per paper (p10 / p50 / p90) | 1 / **10** / 24 |
| With empty title | 212,160 (0.8%) |
| With noise title | 133,000 (0.5%) |
| With IMRAD match | 6,124,693 (23.2%) |
| Papers where ALL titles are missing/noise | 214,164 (9.7%) |

Here, 0.8% and 0.5% are shares of **all sections**, while 9.7% is the share of **all papers** whose section titles are entirely missing or just noise.

**Per-section word count** (estimated from a sample):

| Statistic | Words |
|-----------|------:|
| Mean | **502** |
| p50 (median) | **311** (~404 tokens → fits in one chunk (≤700 tok)) |
| p90 | 1,013 |

### Top 30 Normalised Section Titles

> Titles are lowercased and leading numbers stripped.

| Rank | Title | Count | IMRAD |
|-----:|-------|------:|:-----:|
| 1 | introduction `████████████` | 1,751,045 | ✓ |
| 2 | conclusion `███░░░░░░░░░` | 471,130 | ✓ |
| 3 | conclusions `██░░░░░░░░░░` | 363,967 | ✓ |
| 4 | acknowledgements `██░░░░░░░░░░` | 298,768 | ✓ |
| 5 | acknowledgments `█░░░░░░░░░░░` | 264,492 |  |
| 6 | discussion `█░░░░░░░░░░░` | 209,640 | ✓ |
| 7 | results `█░░░░░░░░░░░` | 176,933 | ✓ |
| 8 | related work `█░░░░░░░░░░░` | 164,023 | ✓ |
| 9 | preliminaries `░░░░░░░░░░░░` | 106,768 |  |
| 10 | proof of theorem `░░░░░░░░░░░░` | 101,651 |  |
| 11 | summary `░░░░░░░░░░░░` | 101,004 |  |
| 12 | experiments `░░░░░░░░░░░░` | 88,636 | ✓ |
| 13 | acknowledgement `░░░░░░░░░░░░` | 67,886 | ✓ |
| 14 | acknowledgment `░░░░░░░░░░░░` | 62,607 |  |
| 15 | concluding remarks `░░░░░░░░░░░░` | 46,510 |  |
| 16 | results and discussion `░░░░░░░░░░░░` | 42,625 | ✓ |
| 17 | appendix `░░░░░░░░░░░░` | 41,161 | ✓ |
| 18 | methodology `░░░░░░░░░░░░` | 40,567 | ✓ |
| 19 | background `░░░░░░░░░░░░` | 40,543 | ✓ |
| 20 | experimental setup `░░░░░░░░░░░░` | 40,152 | ✓ |
| 21 | numerical results `░░░░░░░░░░░░` | 39,955 | ✓ |
| 22 | method `░░░░░░░░░░░░` | 39,143 | ✓ |
| 23 | main results `░░░░░░░░░░░░` | 38,228 | ✓ |
| 24 | implementation details `░░░░░░░░░░░░` | 37,548 |  |
| 25 | datasets `░░░░░░░░░░░░` | 35,755 |  |
| 26 | methods `░░░░░░░░░░░░` | 34,409 | ✓ |
| 27 | notation `░░░░░░░░░░░░` | 33,853 |  |
| 28 | proof of lemma `░░░░░░░░░░░░` | 33,189 |  |
| 29 | experimental results `░░░░░░░░░░░░` | 31,774 | ✓ |
| 30 | summary and conclusions `░░░░░░░░░░░░` | 31,541 | ✓ |

---
##  5 · In-Text Citation Markers

> `{{cite:sha_hex}}` markers in section text point to a key in `bib_entries`.
> We count **unique** ref_ids per paper.

| Metric | Value |
|--------|------:|
| Total marker occurrences | **124,699,637** |
| Avg occurrences / paper | **56.3** |
| Papers with NO citations | 261,136 (11.8%) |
| Unique citations per paper (p50 / p90) | **26** / 63 |
| Figure markers | 10,270,985 |
| Table markers | 3,012,103 |

**Citation concentration**

| Group | Share of total citations |
|-------|------------------------:|
| Top 10% of papers | **32.5%** |
| Top 25% of papers | **57.0%** |

---
##  6 · Bibliography Entry Quality

> **SHA-only** = no external ID exists — the entry is only reachable via its local SHA key.

| Metric | Count | % of entries |
|--------|------:|-------------:|
| Total bib entries | **76,338,716** | — |
| Avg per paper | **34.5** | — |
| Has DOI | 30,138,038 | **39.5%** |
| Has arXiv ID | 6,209,496 | **8.1%** |
| Has OpenAlex ID | 34,073,397 | **44.6%** |
| Has DOI, arXiv, or OpenAlex | 34,073,397 | **44.6%** |
| SHA-only (no external ID) | **42,265,319** | **55.4%** |
| Has title string | 76,326,134 | 100.0% |
| Has year | 0 | 0.0% |

**Field presence across all bib entries**

| Field key | % present |
|-----------|----------:|
| `bib_entry_raw` | 100.0% |
| `ids` | 83.0% |
| `discipline` | 35.7% |
| `contained_links` | 13.3% |
| `contained_arXiv_ids` | 3.4% |

---
##  7 · Cite-Marker → Bib Resolution

> For each unique `{{cite:sha}}` we look up `bib_entries[sha]` and check for a DOI/arXiv ID.

| Path | Count | % |
|------|------:|--:|
| Unique ref_ids total | **69,815,848** | — |
| → resolved via DOI | 27,733,463 | 39.7% |
| → resolved via arXiv ID | 67,514 | 0.1% |
| → resolved via OpenAlex ID | 3,649,807 | 5.2% |
| → SHA-only (API required) | **38,365,064** | **55.0%** |

| Summary | % |
|---------|--:|
| Resolvable without API call | **45.0%** |
| Require Crossref / OpenAlex | **55.0%** |

In other words, 55% of cited references only have a local SHA key in the sections (no DOI/arXiv/OpenAlex ID yet), so we must call external APIs using the title/metadata to attach a public identifier.

---
##  8 · Discipline Breakdown

| arXiv prefix | Count |
|-------------|------:|
| `cs` | 863,182 |
| `math` | 737,033 |
| `cond-mat` | 399,102 |
| `astro-ph` | 382,684 |
| `physics` | 218,873 |
| `hep-ph` | 164,483 |
| `hep-th` | 142,745 |
| `quant-ph` | 126,968 |
| `stat` | 112,104 |
| `gr-qc` | 98,485 |
| `eess` | 75,015 |
| `math-ph` | 70,670 |
| `nucl-th` | 49,354 |
| `hep-ex` | 49,121 |
| `nlin` | 35,697 |

---
##  9 · Chunking Yield Estimate

> Estimates how many Qdrant vectors each paper will produce.

```
tokens  = words × 1.3
chunks  = 1 if tokens <= 700
        = ceil((tokens - 650) / 600) + 1 otherwise
paper   = sum(section_chunks) + 1
```

| Statistic | Chunks / paper |
|-----------|---------------:|
| Mean | **19.7** |
| p10 | 6 |
| p25 | 10 |
| p50 (median) | **16** |
| p75 | 25 |
| p90 | 36 |

---
##  10 · Scale Projections → 2,338,911 papers

> Linear extrapolation from the analysed sample.

### Qdrant Vectors

| Basis | Vectors |
|-------|--------:|
| Mean chunks/paper | **46,159,250** (46.2M) |
| p50 chunks/paper | 37,422,576 (37.4M) |

### Storage (dim=768, +30% HNSW graph overhead)

| Component | float32 | int8 (HPC config) |
|-----------|--------:|------------------:|
| Vectors | 184.3 GB | **46.1 GB** |
| Payload (~800 B/point) | 36.9 GB | 36.9 GB |
| **Total** | **221.3 GB** | **83.0 GB** |

### Bibliography at Scale

| | Count |
|--|------:|
| Total bib entries | 80,640,948 (80.6M) |
| SHA-only needing resolution | **44,647,271** (44.6M) |

### API Resolution Budget

| | |
|--|--|
| Total Crossref/OpenAlex calls needed | **40,527,512** (40.5M) |
| At 50 req/s (OpenAlex polite pool) | **225.2 hours** |
| At 10 req/s (Crossref free tier) | 1125.8 hours |

We handle this with a two-lane design: the ingest pipeline never blocks on APIs (it stores SHA-only refs), while a separate background resolver service slowly enriches all unresolved refs at a controlled request rate.

---
##  11 · Evidence Graph Projection

| Metric | Count | % |
|--------|------:|--:|
| Total citation marker occurrences | 131,960,820 | — |
| Unique cited ref_ids | 73,751,021 | — |
| Unique cited ref_ids resolvable without API | 33,223,508 | 45.0% |
| Unique cited ref_ids needing API resolution | **40,527,512** | **55.0%** |
| Bib entries with DOI | — | 39.5% |
| Bib entries with arXiv ID | — | 8.1% |

---
##  12 · Pipeline Compatibility Summary

| Level | Issue |
|:-----:|-------|
| ⚠️ | Year is 0% in metadata — must derive from arXiv ID YYMM prefix at ingest |
| ⚠️ | 55.4% of bib entries are SHA-only |
| ⚠️ | 55.0% of citations need API → ~40.5M calls at scale |
| ℹ️ | 20 chunks/paper (mean) → ~46M vectors at scale |

---
