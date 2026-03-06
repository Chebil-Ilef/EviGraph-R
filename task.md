curl -G "https://api.crossref.org/works" \
  --data-urlencode "query.title=Attention Is All You Need" \
  -d "rows=3" \
  -d "select=DOI,title,author,published" \
  -H "User-Agent: myapp/1.0 (mailto:you@example.com)"



  curl -G "https://api.openalex.org/works" \
  --data-urlencode "search=Attention Is All You Need" \
  -d "select=id,title,doi,ids" \
  -d "per-page=3" \
  -H "User-Agent: myapp/1.0 (mailto:you@example.com)"



https://export.arxiv.org/api/query?max_results=3&sortBy=relevance&search_query=BERT: Pre-training of Deep Bidirectional Transformers


raw reference
      │
      ▼
title extraction / normalization
      │
      ▼
OpenAlex lookup
      │
      ├── found
      │      ├── DOI ID or/and openalex id or/and arXiv ID present in OpenAlex metadata
      │      │        → store 
      │      │
      │      └── none
      │               → continue
      │
      └── not found
             ▼
        Crossref lookup
             │
             ├── found DOI 
             │      → continue
             │
             └── not found
                    ▼
               arXiv lookup (title search via arXiv API)
                    │
                    ├── found
                    │       → store arXiv_id
                    │
                    └── not found
                           ▼
                      unresolved node




give me pseudo code with which apis to call 

ALSO IF YOU READ ANY RATE LIMIT IN ANY STEP CONTINUE TO NEXT STEP AND IF FINAL THEN UNRESOLVED