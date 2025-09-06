# Usage Examples

## Running Retrieval
```python
from src.retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(sparse_weight=0.3, dense_weight=0.7)
results = retriever.retrieve("What is RAG?", corpus)
```

## Evaluating Hallucinations
```python
from src.generation.hallucination_judge import Judge

judge = Judge(model="gemini")
score = judge.evaluate(answer, context)
```
