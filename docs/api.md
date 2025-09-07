# API Reference

This document provides a comprehensive reference for the Local Hybrid Retrieval-Augmented Document QA API.

## Table of Contents

- [Core Classes](#core-classes)
- [Retrieval API](#retrieval-api)
- [Generation API](#generation-api)
- [Evaluation API](#evaluation-api)
- [Web API](#web-api)

## Core Classes

### HybridRetriever

Main class for hybrid retrieval combining sparse and dense methods.

```python
class HybridRetriever:
    def __init__(self,
                 sparse_weight: float = 0.3,
                 dense_weight: float = 0.7,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 device: str = "auto"):
        """Initialize hybrid retriever.

        Args:
            sparse_weight: Weight for BM25 scores (0.0 to 1.0)
            dense_weight: Weight for embedding scores (0.0 to 1.0)
            embedding_model: Sentence transformer model name
            device: Device for computation ('cpu', 'cuda', 'auto')
        """

    def retrieve(self,
                query: str,
                corpus: List[str],
                top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant documents.

        Args:
            query: Search query string
            corpus: List of document texts
            top_k: Number of results to return

        Returns:
            List of result dictionaries with keys:
            - 'text': Document text
            - 'score': Combined score
            - 'sparse_score': BM25 score
            - 'dense_score': Embedding score
            - 'index': Original corpus index
        """

    def batch_retrieve(self,
                      queries: List[str],
                      corpus: List[str],
                      top_k: int = 10,
                      batch_size: int = 8) -> List[List[Dict[str, Any]]]:
        """Retrieve documents for multiple queries efficiently.

        Args:
            queries: List of query strings
            corpus: List of document texts
            top_k: Number of results per query
            batch_size: Batch size for processing

        Returns:
            List of result lists (one per query)
        """
```

### Judge

Hallucination detection using LLM-as-Judge approach.

```python
class Judge:
    def __init__(self,
                 model: str = "local",
                 threshold: float = 0.3):
        """Initialize hallucination judge.

        Args:
            model: Model type ('local' or 'api')
            threshold: Hallucination threshold (0.0 to 1.0)
        """

    def evaluate(self,
                answer: str,
                context: str,
                question: str = None) -> float:
        """Evaluate answer for hallucinations.

        Args:
            answer: Generated answer text
            context: Source context text
            question: Original question (optional)

        Returns:
            Hallucination score (0.0 = no hallucination, 1.0 = high hallucination)
        """

    def is_hallucination(self, score: float) -> bool:
        """Check if score indicates hallucination.

        Args:
            score: Hallucination score

        Returns:
            True if hallucination detected
        """
```

## Retrieval API

### SparseRetriever (BM25)

```python
class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 retriever.

        Args:
            k1: Term frequency scaling parameter
            b: Document length normalization parameter
        """

    def fit(self, corpus: List[str]) -> None:
        """Fit retriever on corpus.

        Args:
            corpus: List of document texts
        """

    def score(self, query: str, corpus: List[str] = None) -> np.ndarray:
        """Score query against corpus.

        Args:
            query: Query string
            corpus: Optional corpus (uses fitted corpus if None)

        Returns:
            Array of BM25 scores
        """
```

### DenseRetriever (Embeddings)

```python
class DenseRetriever:
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 device: str = "auto"):
        """Initialize dense retriever.

        Args:
            model_name: Sentence transformer model name
            device: Computation device
        """

    def encode_corpus(self, corpus: List[str]) -> None:
        """Encode and index corpus.

        Args:
            corpus: List of document texts
        """

    def score(self, query: str) -> np.ndarray:
        """Score query against indexed corpus.

        Args:
            query: Query string

        Returns:
            Array of cosine similarity scores
        """
```

## Generation API

### Generator

```python
class Generator:
    def __init__(self,
                 model_name: str = "microsoft/DialoGPT-small",
                 device: str = "auto",
                 max_new_tokens: int = 100,
                 temperature: float = 0.7):
        """Initialize text generator.

        Args:
            model_name: Hugging Face model name
            device: Computation device
            max_new_tokens: Maximum response length
            temperature: Sampling temperature
        """

    def generate(self,
                prompt: str,
                context: str = None) -> str:
        """Generate response to prompt.

        Args:
            prompt: Input prompt
            context: Optional context information

        Returns:
            Generated text response
        """

    def generate_with_context(self,
                             question: str,
                             context_docs: List[str],
                             max_context_length: int = 1000) -> str:
        """Generate answer with retrieved context.

        Args:
            question: User question
            context_docs: Retrieved context documents
            max_context_length: Maximum context length

        Returns:
            Generated answer
        """
```

## Evaluation API

### Evaluator

```python
class Evaluator:
    def __init__(self, metrics: List[str] = None):
        """Initialize evaluator.

        Args:
            metrics: List of metrics to compute
                    ['exact_match', 'f1', 'rouge', 'bleu']
        """

    def evaluate(self,
                predictions: List[str],
                ground_truth: List[str],
                questions: List[str] = None) -> Dict[str, float]:
        """Evaluate predictions against ground truth.

        Args:
            predictions: List of predicted answers
            ground_truth: List of correct answers
            questions: List of questions (optional)

        Returns:
            Dictionary of metric scores
        """

    def evaluate_single(self,
                       prediction: str,
                       ground_truth: str,
                       question: str = None) -> Dict[str, float]:
        """Evaluate single prediction.

        Args:
            prediction: Predicted answer
            ground_truth: Correct answer
            question: Question (optional)

        Returns:
            Dictionary of metric scores
        """
```

### BootstrapEvaluator

```python
class BootstrapEvaluator:
    def __init__(self,
                 n_samples: int = 1000,
                 confidence_level: float = 0.95,
                 random_seed: int = 42):
        """Initialize bootstrap evaluator.

        Args:
            n_samples: Number of bootstrap samples
            confidence_level: Confidence level (0.0 to 1.0)
            random_seed: Random seed for reproducibility
        """

    def compute_confidence_intervals(self,
                                   scores: List[float],
                                   metric_name: str = "accuracy") -> Tuple[float, float]:
        """Compute confidence intervals using bootstrap.

        Args:
            scores: List of individual scores
            metric_name: Name of metric for logging

        Returns:
            Tuple of (lower_bound, upper_bound)
        """

    def evaluate_with_ci(self,
                        predictions: List[str],
                        ground_truth: List[str]) -> Dict[str, Any]:
        """Evaluate with confidence intervals.

        Args:
            predictions: List of predictions
            ground_truth: List of ground truth answers

        Returns:
            Dictionary with scores and confidence intervals
        """
```

## Web API

### REST Endpoints

#### POST /upload

Upload documents for processing.

**Request**:

```json
{
  "files": ["document1.pdf", "document2.txt"],
  "options": {
    "chunk_size": 500,
    "overlap": 50
  }
}
```

**Response**:

```json
{
  "status": "success",
  "document_count": 2,
  "total_chunks": 45
}
```

#### POST /query

Ask questions about uploaded documents.

**Request**:

```json
{
  "question": "What is machine learning?",
  "top_k": 5,
  "include_context": true
}
```

**Response**:

```json
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "confidence": 0.87,
  "hallucination_score": 0.12,
  "sources": [
    {
      "text": "Machine learning uses algorithms...",
      "score": 0.92,
      "document": "ml_intro.pdf"
    }
  ]
}
```

#### GET /status

Get system status and configuration.

**Response**:

```json
{
  "status": "ready",
  "version": "1.0.0",
  "documents_loaded": 5,
  "model_loaded": true,
  "memory_usage": "2.1GB"
}
```

#### GET /config

Get current system configuration.

**Response**:

```json
{
  "retrieval": {
    "sparse_weight": 0.3,
    "dense_weight": 0.7,
    "top_k": 10
  },
  "generation": {
    "model": "microsoft/DialoGPT-small",
    "max_tokens": 100
  }
}
```

#### POST /config

Update system configuration.

**Request**:

```json
{
  "retrieval": {
    "sparse_weight": 0.4,
    "dense_weight": 0.6
  }
}
```

**Response**:

```json
{
  "status": "updated",
  "message": "Configuration updated successfully"
}
```

### Error Responses

All endpoints return errors in the following format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid file format",
    "details": {
      "allowed_formats": ["pdf", "txt", "md"],
      "received_format": "exe"
    }
  }
}
```

### Common Error Codes

- `VALIDATION_ERROR`: Invalid input data
- `PROCESSING_ERROR`: Document processing failed
- `MODEL_ERROR`: Model loading or inference error
- `CONFIGURATION_ERROR`: Invalid configuration
- `INTERNAL_ERROR`: Unexpected server error

## Configuration

### Configuration Classes

```python
@dataclass
class RetrievalConfig:
    sparse_weight: float = 0.3
    dense_weight: float = 0.7
    top_k: int = 10
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50

@dataclass
class GenerationConfig:
    model_name: str = "microsoft/DialoGPT-small"
    max_new_tokens: int = 100
    temperature: float = 0.7
    do_sample: bool = True
    repetition_penalty: float = 1.1

@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    max_upload_size: int = 100 * 1024 * 1024  # 100MB

@dataclass
class EvaluationConfig:
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    random_seed: int = 42
    metrics: List[str] = None
```

### Environment Variables

```bash
# Model and data paths
TRANSFORMERS_CACHE=/path/to/models
SENTENCE_TRANSFORMERS_HOME=/path/to/sentence_models

# Performance settings
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4

# Web server settings
FLASK_ENV=production
MAX_CONTENT_LENGTH=104857600

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/system.log
LOG_FORMAT=json
```

## Examples

### Complete Retrieval Pipeline

```python
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.generator import Generator
from src.generation.hallucination_judge import Judge

# Initialize components
retriever = HybridRetriever(sparse_weight=0.4, dense_weight=0.6)
generator = Generator()
judge = Judge()

# Prepare corpus
corpus = ["Document 1 text...", "Document 2 text..."]

# Process query
query = "What is artificial intelligence?"

# Step 1: Retrieve relevant documents
results = retriever.retrieve(query, corpus, top_k=3)
context_docs = [result['text'] for result in results]

# Step 2: Generate answer
answer = generator.generate_with_context(query, context_docs)

# Step 3: Check for hallucinations
context = " ".join(context_docs)
hallucination_score = judge.evaluate(answer, context, query)

# Step 4: Filter if necessary
if hallucination_score > 0.3:
    answer = "I cannot provide a reliable answer based on the available information."

print(f"Answer: {answer}")
print(f"Hallucination Score: {hallucination_score:.3f}")
```

### Batch Processing

```python
# Process multiple queries efficiently
queries = [
    "What is machine learning?",
    "Explain neural networks",
    "What is natural language processing?"
]

batch_results = retriever.batch_retrieve(queries, corpus, top_k=5)

for query, results in zip(queries, batch_results):
    print(f"\nQuery: {query}")
    for i, result in enumerate(results[:3]):  # Top 3 results
        print(f"{i+1}. {result['text'][:100]}...")
        print(f"   Combined Score: {result['score']:.3f}")
```

This API reference covers the main components and usage patterns. For more detailed examples, see the [Usage Guide](usage.md).
