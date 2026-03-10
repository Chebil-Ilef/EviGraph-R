

##  DONE

* create sample from unarxiv
* create batches from a sample
* documentation for startgies and for pipeline in HPC
* resolve_title for missing citation IDs , crutial for later building the graph
* chunker with window overlap
* embedder and indexer
* full pipeline code (orchestrator chunker (from folder batches)- embedder - indexer to qdrant)

## TO DO

* automated script to run qdrant docker if local if down and singularity if hpc if down
* mount volumes to that container

* retrieval script to test HYBRID
* test Dense + BM25 text search + RRF VS Dense + Sparse vectors in Qdrant (best with BGE-M3) on a small sample 
