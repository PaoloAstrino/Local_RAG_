# API Reference

This document provides a comprehensive reference for the Local Hybrid Retrieval-Augmented Document QA system.

## Table of Contents

- [Socket Communication Protocol](#socket-communication-protocol)
- [Server Commands](#server-commands)
- [HTTP API Endpoints](#http-api-endpoints)
- [Configuration](#configuration)

## Socket Communication Protocol

The system uses JSON-based commands over TCP sockets for client-server communication.

### Connection Parameters

```python
HOST = '127.0.0.1'  # Local server
PORT = 65432         # Server port
```

### Command Structure

```python
{
    "command": str,      # Command name
    # Additional parameters based on command
}
```

### Response Structure

```python
{
    "status": str,       # "success" or "error"
    "message": str,      # Optional message
    # Additional data based on command
}
```

## Server Commands

### upload

Upload and index documents for retrieval.

```python
command = {
    "command": "upload",
    "file_paths": [str]  # List of absolute file paths
}

response = {
    "status": "success",
    "message": "Files uploaded and retriever initialized successfully",
    "files": [str]  # List of successfully uploaded filenames
}
```

**Supported formats**: PDF, CSV, JSON, TXT

**Process**:

1. Validates file types and sizes
2. Loads documents via specialized loaders
3. Chunks text (400 chars, 150 overlap)
4. Generates BGE embeddings (cached)
5. Builds hybrid retriever (30% BM25, 70% semantic)

### ask

Query the system with a question.

```python
command = {
    "command": "ask",
    "question": str  # User question
}

response = {
    "status": "success",
    "answer": str  # Generated answer from Llama 3.2
}
```

**Process**:

1. Retrieves top-5 relevant chunks (hybrid retrieval)
2. Constructs prompt with chat history + context + question
3. Calls Ollama API (Llama 3.2) locally
4. Returns answer and updates chat history

### get_files

List currently uploaded files.

```python
command = {
    "command": "get_files"
}

response = {
    "status": "success",
    "files": [str]  # List of uploaded filenames
}
```

### get_chat_history

Retrieve conversation history.

```python
command = {
    "command": "get_chat_history"
}

response = {
    "status": "success",
    "chat_history": [
        (question: str, answer: str, retrieved_docs: List[str])
    ]
}
```

### get_status

Check system status.

```python
command = {
    "command": "get_status"
}

response = {
    "status": "success",
    "num_files": int,
    "files": [str],
    "qa_chain_initialized": bool
}
```

### reset

Clear all data and reinitialize system.

```python
command = {
    "command": "reset"
}

response = {
    "status": "success",
    "message": "System reset; all files and QA system cleared",
    "files": []
}
```

### delete_file

Remove a specific file.

```python
command = {
    "command": "delete_file",
    "filename": str
}

response = {
    "status": "success",
    "message": "File 'filename' deleted",
    "files": [str]  # Remaining files
}
```

### get_document_content

Retrieve raw content of an uploaded file.

```python
command = {
    "command": "get_document_content",
    "filename": str
}

response = {
    "status": "success",
    "filename": str,
    "content": str  # Full document text
}
```

## HTTP API Endpoints

The Flask client (`src/client.py`) exposes HTTP endpoints that forward to the server.

### POST /rag_upload

Upload documents.

**Request**:

```http
POST /rag_upload HTTP/1.1
Content-Type: multipart/form-data

files: [file1.pdf, file2.csv, ...]
```

**Response**:

```json
{
  "status": "success",
  "message": "Files uploaded and retriever initialized successfully",
  "files": ["file1.pdf", "file2.csv"]
}
```

### POST /rag_ask

Ask a question.

**Request**:

```http
POST /rag_ask HTTP/1.1
Content-Type: application/json

{
  "question": "What is the main topic?"
}
```

**Response**:

```json
{
  "status": "success",
  "answer": "The main topic is..."
}
```

### GET /rag_get_files

List uploaded files.

**Response**:

```json
{
  "status": "success",
  "files": ["file1.pdf", "file2.csv"]
}
```

### GET /rag_get_chat_history

Get conversation history.

**Response**:

```json
{
  "status": "success",
  "chat_history": [
    ["Question 1?", "Answer 1", ["chunk1", "chunk2"]],
    ["Question 2?", "Answer 2", ["chunk3", "chunk4"]]
  ]
}
```

### POST /rag_reset

Reset system.

**Response**:

```json
{
  "status": "success",
  "message": "System reset; all files and QA system cleared"
}
```

### GET /rag_get_status

Check system status.

**Response**:

```json
{
  "status": "success",
  "num_files": 2,
  "files": ["file1.pdf", "file2.csv"],
  "qa_chain_initialized": true
}
```

## Configuration

### Hybrid Retriever Weights

Edit `src/server.py`, function `load_db`:

```python
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.3, 0.7]  # [sparse, dense] - optimal from paper
)
```

### Ollama Settings

Edit `src/server.py`, function `call_external_llm_api`:

```python
ollama_url = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api/generate')

payload = {
    "model": "llama3.2",
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "num_predict": 1024
    }
}
```

### Document Chunking

Edit `src/server.py`, function `load_db`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,      # Characters per chunk
    chunk_overlap=150    # Overlap between chunks
)
```

### Embedding Model

Edit `src/server.py`, function `get_embeddings`:

```python
_underlying_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",  # BGE embeddings
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
)
```

### Environment Variables

Create a `.env` file for optional configuration:

```bash
# Ollama API endpoint (default: http://localhost:11434/api/generate)
OLLAMA_API_URL=http://localhost:11434/api/generate

# Server socket configuration
LOCAL_SERVER_HOST=127.0.0.1
LOCAL_SERVER_PORT=65432

# Client configuration
CLIENT_PORT=5001
```

## Error Handling

All responses include a `status` field:

- `"success"`: Operation completed successfully
- `"error"`: Operation failed, check `message` field for details

Example error response:

```json
{
  "status": "error",
  "message": "Retriever not initialized. Upload files first."
}
```

Common errors:

- **"No files provided"**: Upload command with empty file_paths
- **"Retriever not initialized"**: Asking questions before uploading documents
- **"File not found"**: Referencing non-existent file
- **"Ollama API call failed"**: Ollama not running or model not available
- **"Failed to initialize retriever"**: Document processing error (check logs)
  class DenseRetriever:
  def **init**(self,
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

````

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
````

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
