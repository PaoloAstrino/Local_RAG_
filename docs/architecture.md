# Architecture Overview

The system uses a client-server model with hybrid retrieval:
- **Retrieval**: BM25 + embeddings (via FAISS)
- **Generation**: Local LLM with hallucination filtering
- **Evaluation**: Bootstrap resampling for confidence intervals

See `papers/images/architecture_simple_diagram.png` for a visual.
