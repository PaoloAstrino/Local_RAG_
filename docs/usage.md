# Usage Guide

This guide provides examples and instructions for using the Local Hybrid Retrieval-Augmented Document QA system.

## Table of Contents

- [Quick Start](#quick-start)
- [Web Interface](#web-interface)
- [Python API](#python-api)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Advanced Usage](#advanced-usage)

## Quick Start

### Web Interface

1. **Start the server**:

   ```bash
   python src/server_MPC.py
   ```

2. **Open your browser** to `http://localhost:5000`

3. **Upload documents** and ask questions through the web interface

### Command Line

```bash
# Basic retrieval example
python -c "
from src.retrieval.hybrid_retriever import HybridRetriever
retriever = HybridRetriever(sparse_weight=0.3, dense_weight=0.7)
results = retriever.retrieve('What is machine learning?', corpus)
print(results)
"
```

## Web Interface

### Features

- **Document Upload**: Support for PDF, TXT, and other text formats
- **Real-time QA**: Ask questions about uploaded documents
- **Confidence Scores**: View retrieval and generation confidence
- **Hallucination Detection**: Automatic filtering of unreliable responses

### Usage Steps

1. **Upload Documents**: Click "Choose Files" and select your documents
2. **Ask Questions**: Type your question in the input field
3. **View Results**: See the answer with confidence scores and source references
4. **Adjust Settings**: Modify retrieval weights and other parameters

## Python API

### Basic Retrieval

```python
from src.retrieval.hybrid_retriever import HybridRetriever

# Initialize retriever with custom weights
retriever = HybridRetriever(
    sparse_weight=0.4,
    dense_weight=0.6,
    embedding_model="all-MiniLM-L6-v2"
)

# Prepare your corpus
corpus = [
    "Machine learning is a subset of artificial intelligence...",
    "Deep learning uses neural networks with multiple layers...",
    "Natural language processing handles human language..."
]

# Perform retrieval
query = "What is deep learning?"
results = retriever.retrieve(query, corpus, top_k=5)

for i, result in enumerate(results):
    print(f"{i+1}. {result['text']}")
    print(f"   Score: {result['score']:.3f}")
    print(f"   Sparse: {result['sparse_score']:.3f}")
    print(f"   Dense: {result['dense_score']:.3f}")
    print()
```

### Hallucination Detection

```python
from src.generation.hallucination_judge import Judge

# Initialize judge
judge = Judge(model="local")  # or "api" for external models

# Evaluate answer quality
question = "What is the capital of France?"
answer = "Paris is the capital of France."
context = "France is a country in Europe. Its capital is Paris."

score = judge.evaluate(answer, context, question)
print(f"Hallucination Score: {score:.3f}")  # Lower is better
print(f"Confident: {score < 0.3}")
```

### Custom Configuration

```python
from src.config import Config

# Load default configuration
config = Config()

# Modify settings
config.SPARSE_WEIGHT = 0.5
config.DENSE_WEIGHT = 0.5
config.BATCH_SIZE = 16
config.MAX_LENGTH = 256

# Save configuration
config.save("my_config.json")
```

## Configuration

### Key Parameters

#### Retrieval Settings

```python
SPARSE_WEIGHT = 0.3          # Weight for BM25 (0.0 to 1.0)
DENSE_WEIGHT = 0.7           # Weight for embeddings (0.0 to 1.0)
TOP_K = 10                   # Number of results to return
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
```

#### Generation Settings

```python
GENERATION_MODEL = "microsoft/DialoGPT-small"  # Local model path
MAX_NEW_TOKENS = 100         # Maximum response length
TEMPERATURE = 0.7            # Response randomness (0.0 to 1.0)
DO_SAMPLE = True             # Enable sampling
```

#### Evaluation Settings

```python
BOOTSTRAP_SAMPLES = 1000     # Bootstrap resampling iterations
CONFIDENCE_LEVEL = 0.95      # Confidence interval level
RANDOM_SEED = 42             # Random seed for reproducibility
```

### Environment Variables

```bash
# Model cache directory
export TRANSFORMERS_CACHE=/path/to/cache

# CUDA settings
export CUDA_VISIBLE_DEVICES=0

# Logging level
export LOG_LEVEL=INFO
```

## Evaluation

### Running Evaluations

#### Single Dataset

```bash
python src/evaluation/evaluate_dataset.py \
    --dataset squad \
    --sparse_weight 0.3 \
    --dense_weight 0.7 \
    --output results/squad_results.json
```

#### Multiple Datasets

```bash
python src/evaluation/run_weight_sweep.py \
    --datasets squad marco natural_questions \
    --weight_range 0.0 1.0 0.1 \
    --bootstrap_samples 1000
```

#### Custom Evaluation

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator()
results = evaluator.evaluate(
    predictions=your_predictions,
    ground_truth=your_ground_truth,
    metrics=['exact_match', 'f1', 'rouge']
)

print(f"Exact Match: {results['exact_match']:.3f}")
print(f"F1 Score: {results['f1']:.3f}")
```

### Evaluation Metrics

- **Exact Match**: Percentage of questions answered exactly correctly
- **F1 Score**: Token-level F1 score for partial matches
- **ROUGE**: Recall-oriented Understudy for Gisting Evaluation
- **BLEU**: Bilingual Evaluation Understudy for machine translation
- **Hallucination Rate**: Percentage of responses with detected hallucinations

### Bootstrap Confidence Intervals

```python
from src.evaluation.bootstrap import BootstrapEvaluator

bootstrap_eval = BootstrapEvaluator(n_samples=1000)
confidence_intervals = bootstrap_eval.compute_confidence_intervals(
    scores=your_scores,
    confidence_level=0.95
)

print(f"95% CI: [{confidence_intervals[0]:.3f}, {confidence_intervals[1]:.3f}]")
```

## Advanced Usage

### Custom Embedding Models

```python
from sentence_transformers import SentenceTransformer
from src.retrieval.dense_retriever import DenseRetriever

# Load custom model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
retriever = DenseRetriever(model=model)

# Use in hybrid setup
hybrid = HybridRetriever(
    sparse_retriever=BM25Retriever(),
    dense_retriever=retriever,
    sparse_weight=0.4,
    dense_weight=0.6
)
```

### Batch Processing

```python
# Process multiple queries efficiently
queries = [
    "What is machine learning?",
    "Explain neural networks",
    "What is natural language processing?"
]

batch_results = retriever.batch_retrieve(queries, corpus, batch_size=8)

for query, results in zip(queries, batch_results):
    print(f"Query: {query}")
    print(f"Top result: {results[0]['text'][:100]}...")
    print()
```

### Custom Document Processing

```python
from src.utils.document_processor import DocumentProcessor

processor = DocumentProcessor()

# Process different file types
documents = processor.process_files([
    "document.pdf",
    "article.txt",
    "webpage.html"
])

# Extract metadata
for doc in documents:
    print(f"Title: {doc['title']}")
    print(f"Content length: {len(doc['content'])}")
    print(f"Source: {doc['source']}")
```

### Performance Monitoring

```python
import time
from src.utils.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()

with monitor.track_operation("retrieval"):
    results = retriever.retrieve(query, corpus)

print(f"Retrieval time: {monitor.get_last_duration():.3f}s")
print(f"Memory usage: {monitor.get_memory_usage():.2f}MB")
```

## Troubleshooting

### Common Issues

#### Slow Performance

- Reduce `TOP_K` in configuration
- Use smaller embedding models
- Enable GPU acceleration if available
- Process documents in smaller batches

#### Memory Errors

- Reduce `BATCH_SIZE` in configuration
- Process fewer documents at once
- Use CPU-only mode for large datasets
- Clear cache periodically

#### Low Quality Results

- Adjust sparse/dense weights
- Try different embedding models
- Increase `TOP_K` for more candidates
- Fine-tune generation parameters

### Getting Help

- Check the [GitHub Issues](https://github.com/username/repo/issues) page
- Review the [Architecture Guide](architecture.md) for technical details
- Run the test suite: `python -m pytest tests/`
- Enable debug logging: `export LOG_LEVEL=DEBUG`
