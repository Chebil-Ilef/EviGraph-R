## DONE

- Sample created from UnarXiv dataset
- Sample split into processing batches
- Documentation for pipeline strategy and HPC execution
- `resolve_title` implemented to recover missing citation IDs (needed for graph construction)
- Chunker implemented with sliding window overlap
- Embedder and indexer implemented
- Full pipeline implemented (orchestrator → chunker → embedder → Qdrant indexer)
- Retrieval script implemented for hybrid retrieval experiments
- Experiments comparing Dense + BM25 (RRF) vs Dense + Sparse (BGE-M3) on a small sample
- Codebase refactor to support LangGraph + DSPy + Pydantic architecture
- Re-test full indexing pipeline end-to-end
- Implement LangGraph StateGraph orchestration
- Implement Agent 1: query decomposer
- report about unarxiv dataset
- are there are datasource to enrich with ?
- align code with architecture
- Implement hybrid retrieval node
- Experiment with Agent 2: claims graph builder after reading from litterature
- two post indexing pipelines: add ids for references and enhance imrad section titles
- Implement Agent 3: reasoning judge
- Implement Agent 4: answer generator with citations


## TODO

- re-test indexing + two post indexing scripts
- fix issues with graph 
- visualize and test every step of the architecture

- add OpenTelemetry or Langfuse as observability tools
- Run full-scale experiments on HPC (indexing + system pipeline)
- Publish dense and sparse indices on HuggingFace

- Perform full system evaluation

- Explore packaging options (pip package, Docker MCP server, open-source UI)
- Implement auto-start script for Qdrant (Docker locally, Singularity on HPC)
- Configure persistent volume mounts for vector database

## ANY RANDOM IDEA