import os
import sys
import json
import pandas as pd
import time
import socket
import threading
from typing import List, Dict, Any
# Updated import for Document
from langchain_core.documents import Document
from jsonschema import validate, ValidationError
import requests  # <-- Add this import for API calls
import torch # <-- Import torch

# Import LangChain components
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.storage import LocalFileStore # <-- Import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings # <-- Import CacheBackedEmbeddings
from langchain_community.retrievers import BM25Retriever # <-- Updated import path
from langchain.retrievers.ensemble import EnsembleRetriever # <-- Updated import path

# Document loaders for different file types
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader

from dotenv import load_dotenv
load_dotenv()

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
logger = logging.getLogger(__name__)

def sanitize_filename(filename: str) -> str:
    # Remove path separators and suspicious patterns
    filename = os.path.basename(filename)
    if '..' in filename or filename.startswith(('/', '\\')):
        raise ValueError('Suspicious filename detected')
    return filename


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'csv', 'json', 'txt'} # <-- Added txt support
# Increase max file size limit (e.g., 500MB) - adjust as needed
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

stop_generation_flag = [False]

uploaded_files = []
# --- Move embeddings initialization to function to avoid import issues ---
def get_embeddings():
    """Lazy initialization of embeddings to avoid TensorFlow import issues during startup."""
    global _underlying_embeddings, embeddings
    if '_underlying_embeddings' not in globals():
        logger.info("Initializing embeddings...")
        _underlying_embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

        # --- Set up cache ---
        EMBED_CACHE_DIR = "./embed_cache/"
        os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
        store = LocalFileStore(EMBED_CACHE_DIR)

        # --- Create CacheBackedEmbeddings ---
        embeddings = CacheBackedEmbeddings.from_bytes_store(
            _underlying_embeddings, store, namespace=_underlying_embeddings.model_name
        )
        logger.info("Embeddings initialized successfully.")
    return embeddings
# --------------------------------------

retriever = None
chat_history = []  # Now stores (question, answer, retrieved_docs) tuples

def extract_answer(text):
    parts = text.split("Answer:", 1)
    return parts[1].strip() if len(parts) > 1 else text

def get_prompt_template():
    # Updated template to include chat history
    return """You are a helpful assistant. Based on the chat history and the context provided, answer the user's current question in a clear and natural way.
Summarize the key information relevant to the question without using bullet points unless necessary for clarity.
If you don't know the answer or the context doesn't contain the information, just say so.
Try to sound helpful and conversational.

Chat History:
{chat_history}

Context:
{context}

Current Question: {question}

Answer:"""

def format_chat_history(history: List[tuple]) -> str:
    """Formats chat history into a readable string for the LLM."""
    if not history:
        return "No previous conversation."
    formatted_history = []
    for entry in history:
        # Handle both old (question, answer) and new (question, answer, retrieved_docs) formats
        if len(entry) >= 2:
            user_msg, assistant_msg = entry[0], entry[1]
            formatted_history.append(f"Human: {user_msg}")
            formatted_history.append(f"Assistant: {assistant_msg}")
    return "\n".join(formatted_history)

def construct_llm_payload(question: str, context: str, history: List) -> Dict[str, Any]:
    """Construct the payload for Ollama LLM API call."""
    prompt_template = get_prompt_template()
    formatted_history = format_chat_history(history)
    prompt = prompt_template.format(chat_history=formatted_history, context=context, question=question)
    return {
        "model": "llama3.2",  # Use Llama 3.2 model from Ollama
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 40,
            "num_predict": 10024,
        }
    }

def handle_llm_response(response: requests.Response) -> str:
    """Handle the response from Ollama LLM API."""
    if response.status_code == 200:
        data = response.json()
        try:
            return data['response']
        except Exception:
            return "No answer returned from Ollama API."
    else:
        logger.error(f"Ollama API error: {response.status_code} {response.text}")
        return f"Ollama API error: {response.status_code} {response.text}"

def call_external_llm_api(question: str, context: str, history: List) -> str:
    """Call Ollama LLM API with proper error handling."""
    # Ollama API endpoint (default localhost:11434)
    ollama_url = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api/generate')
    
    payload = construct_llm_payload(question, context, history)
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(ollama_url, json=payload, headers=headers, timeout=60)  # Longer timeout for local LLM
        return handle_llm_response(response)
    except Exception as e:
        return f"Ollama API call failed: {str(e)}"

def load_pdf_document(file_path: str) -> List['Document']:
    """Load a PDF document."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages from PDF {file_path}")
    if docs:
        logger.info(f"First page content length: {len(docs[0].page_content)}")
        logger.info(f"First page preview: {docs[0].page_content[:200]}...")
    return docs

def load_csv_document(file_path: str) -> List['Document']:
    """Load a CSV document by converting to string."""
    df = pd.read_csv(file_path)
    content = df.to_string(index=False)
    return [Document(page_content=content, metadata={'source': file_path})]

def load_json_document(file_path: str) -> List['Document']:
    """Load a JSON document."""
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON file: {file_path}")
            return []

    # Simplified JSON loading assuming content is text or can be stringified
    if isinstance(data, list):
        content = "\n".join([json.dumps(item) for item in data])
    elif isinstance(data, dict):
        content = json.dumps(data)
    else:
        content = str(data) # Fallback for other JSON structures
    return [Document(page_content=content, metadata={'source': file_path})]

def load_txt_document(file_path: str) -> List['Document']:
    """Load a text document."""
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()

def load_single_document(file_path: str, file_extension: str) -> List['Document']:
    """Load a single document based on file extension."""
    try:
        if file_extension == 'pdf':
            return load_pdf_document(file_path)
        elif file_extension == 'csv':
            return load_csv_document(file_path)
        elif file_extension == 'json':
            return load_json_document(file_path)
        elif file_extension == 'txt':
            return load_txt_document(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_extension} for file {file_path}")
            return []
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}", exc_info=True)
        return []

def load_documents(file_paths: List[str]) -> List['Document']:
    """Load documents from multiple file paths with proper error handling."""
    documents = []
    for file_path in file_paths:
        file_extension = file_path.split('.')[-1].lower()
        try:
            # Sanitize filename before loading
            safe_path = os.path.join(UPLOAD_FOLDER, sanitize_filename(os.path.basename(file_path)))
            if not os.path.exists(safe_path):
                 logger.warning(f"File not found during loading: {safe_path}")
                 continue # Skip if file doesn't exist at the expected location

            loaded_documents = load_single_document(safe_path, file_extension)

            # Add metadata
            for doc in loaded_documents:
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['file_path'] = safe_path
            documents.extend(loaded_documents)
        except Exception as e:
            logger.error(f"Error loading or processing {file_path} (path used: {safe_path}): {e}", exc_info=True)
    return documents

def load_db(file_paths, k=5):
    documents = load_documents(file_paths)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    # Updated import for DocArrayInMemorySearch
    from langchain_community.vectorstores import DocArrayInMemorySearch
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    if not docs:
        raise ValueError("No document content could be extracted")
    logger.info(f"Creating vector store and BM25 index with {len(docs)} document chunks.")

    # Get embeddings lazily
    embeddings = get_embeddings()

    # 1. Semantic Retriever (using cached embeddings)
    logger.debug("Initializing semantic retriever (DocArrayInMemorySearch)...")
    semantic_db = DocArrayInMemorySearch.from_documents(docs, embeddings) # Use cached embeddings
    semantic_retriever = semantic_db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    logger.debug("Semantic retriever initialized.")

    # 2. Keyword Retriever (BM25)
    logger.debug("Initializing keyword retriever (BM25)...")
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k # Set k for BM25
    logger.debug("Keyword retriever initialized.")

    # 3. Ensemble Retriever
    logger.debug("Initializing Ensemble Retriever...")
    # Combine semantic and keyword retrievers. Adjust weights as needed.
    # Weights determine the contribution of each retriever's scores.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6] # Example: 40% BM25, 60% Semantic
    )
    logger.debug("Ensemble Retriever initialized.")

    # The global 'retriever' will now be the EnsembleRetriever
    return ensemble_retriever

def initialize_qa_system(file_paths):
    global retriever
    logger.info(f"Starting QA system initialization with {len(file_paths)} files...")
    start_time = time.time()
    try:
        retriever = load_db(file_paths)
        end_time = time.time()
        logger.info(f"Successfully initialized QA system in {end_time - start_time:.2f} seconds.")
        return True
    except Exception as e:
        end_time = time.time()
        logger.error(f"Error initializing retriever after {end_time - start_time:.2f} seconds: {e}", exc_info=True) # Add exc_info
        retriever = None # Ensure retriever is None on failure
        return False

def validate_file(file_path: str) -> bool:
    try:
        file_extension = file_path.split('.')[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Invalid file type: {file_extension}")
        if os.path.getsize(file_path) > MAX_FILE_SIZE:
            raise ValueError("File size exceeds the limit")
        # Sanitize filename
        _ = sanitize_filename(file_path)
        return True
    except Exception as e:
        logger.error(f"File validation error: {e}")
        return False

def handle_upload_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    """Handle file upload command with validation and QA system initialization."""
    file_paths = cmd.get("file_paths", [])
    if not file_paths:
        logger.warning("Upload command received with no file paths.")
        return create_error_response("No files provided")

    logger.info(f"Validating {len(file_paths)} files for upload...")
    # Validate files *before* attempting initialization
    valid_files_info = []
    invalid_files_info = []
    for fp in file_paths:
         try:
             # Use sanitize_filename within validate_file for consistency
             if validate_file(fp):
                 valid_files_info.append(os.path.basename(fp))
             else:
                 # If validate_file returns False, log it
                 logger.warning(f"Validation failed for file: {fp}")
                 invalid_files_info.append(os.path.basename(fp) + " (validation failed)")
         except Exception as val_e:
             # Catch potential errors during validation itself
             logger.error(f"Error validating file {fp}: {val_e}")
             invalid_files_info.append(os.path.basename(fp) + f" (validation error: {val_e})")

    if invalid_files_info:
         logger.error(f"Upload failed due to invalid files: {invalid_files_info}")
         # Return specific error about invalid files
         return create_error_response(f"Validation failed for files: {', '.join(invalid_files_info)}")

    # Proceed only with valid file paths corresponding to the validated names
    valid_file_paths = [os.path.join(UPLOAD_FOLDER, fname) for fname in valid_files_info]
    file_names = valid_files_info # Use the validated names

    logger.info(f"Starting QA system initialization with {len(valid_file_paths)} valid files...")
    start_time = time.time()
    success = initialize_qa_system(valid_file_paths) # Pass full paths
    end_time = time.time()
    logger.info(f"QA system initialization completed in {end_time - start_time:.2f} seconds, success: {success}")

    if success:
        uploaded_files = file_names # Store only basenames
        logger.info(f"Upload successful. Current files: {uploaded_files}")
        return create_success_response(
            message="Files uploaded and retriever initialized successfully",
            files=file_names
        )
    else:
        # Initialization failed, error already logged in initialize_qa_system
        logger.error("Upload command failed due to QA system initialization error.")
        return create_error_response("Failed to initialize retriever with uploaded files. Check server logs for details.")

def handle_ask_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    """Handle question asking command with retrieval and LLM response."""
    query = cmd.get("question", "")
    if retriever is None:
        return {"status": "error", "message": "Retriever not initialized. Upload files first.", "answer": ""}
    try:
        logger.info(f"Retrieving documents for query: {query}")
        docs = retriever.get_relevant_documents(query)
        logger.info(f"Retrieved {len(docs)} documents")
        if docs:
            logger.info(f"First doc preview: {docs[0].page_content[:200]}...")
        else:
            logger.info("No documents retrieved")
        context = "\n\n".join([doc.page_content for doc in docs])
        logger.info(f"Context length: {len(context)}")
        logger.info(f"Calling LLM API with query, context, and history (length: {len(chat_history)} turns).")
        # Pass the current chat_history to the LLM call
        answer = call_external_llm_api(query, context, chat_history)
        # Store the retrieved documents along with the question and answer
        retrieved_docs = [doc.page_content for doc in docs]
        chat_history.append((query, answer, retrieved_docs))
        logger.info("Successfully received answer and updated chat history with retrieved documents.")
        return {"status": "success", "answer": answer}
    except Exception as e:
        logger.error(f"Error during 'ask' command processing: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "answer": ""}

def handle_delete_file_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    """Handle file deletion command."""
    filename = cmd.get("filename")
    if not filename:
        return {"status": "error", "message": "No filename provided"}
    safe_filename = sanitize_filename(filename)
    file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    if safe_filename not in uploaded_files:
        return {"status": "error", "message": "File not found in uploaded files"}
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        uploaded_files.remove(safe_filename)
        # Re-initialize QA system with remaining files
        if uploaded_files:
            initialize_qa_system([os.path.join(UPLOAD_FOLDER, f) for f in uploaded_files])
        else:
            retriever = None
        return {"status": "success", "message": f"File '{safe_filename}' deleted", "files": uploaded_files}
    except Exception as e:
        return {"status": "error", "message": f"Failed to delete file: {e}"}

def handle_reset_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    """Handle system reset command."""
    errors = []
    for fname in uploaded_files:
        try:
            fpath = os.path.join(UPLOAD_FOLDER, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        except Exception as e:
            errors.append(str(e))
    uploaded_files.clear()
    retriever = None
    if errors:
        return {"status": "error", "message": "Some files could not be deleted: " + "; ".join(errors)}
    return {"status": "success", "message": "System reset; all files and QA system cleared", "files": []}

def handle_get_document_content_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    """Handle document content retrieval command."""
    filename = cmd.get("filename")
    if not filename:
        return {"status": "error", "message": "No filename provided"}
    try: # Wrap in try-except for sanitization errors
        safe_filename = sanitize_filename(filename)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    if not os.path.exists(file_path):
        # Check if it's in uploaded_files list but physically missing
        if safe_filename in uploaded_files:
             logger.warning(f"File '{safe_filename}' is in the list but not found at '{file_path}'. It might have been deleted externally.")
             # Optionally remove from uploaded_files list here
             # uploaded_files.remove(safe_filename)
             # Consider re-initializing retriever if consistency is critical
        return {"status": "error", "message": f"File not found at expected location: {file_path}"}

    try:
        file_extension = safe_filename.split('.')[-1].lower()
        content = "" # Initialize content

        if file_extension == "pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            content = "\n\n".join([doc.page_content for doc in docs])
        elif file_extension == "csv":
            # Read CSV as plain text
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        elif file_extension == "json":
            # Read JSON as plain text
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        elif file_extension == "txt":
            # Read TXT as plain text
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            # This case should ideally not be reached if ALLOWED_EXTENSIONS is enforced
            return {"status": "error", "message": f"Unsupported file type for content retrieval: {file_extension}"}

        # Log success before returning
        logger.info(f"Successfully retrieved content for file: {safe_filename}")
        return {"status": "success", "filename": safe_filename, "content": content}
    except Exception as e:
        logger.error(f"Failed to read file content for {safe_filename}: {e}", exc_info=True)
        return {"status": "error", "message": f"Failed to read file: {e}"}

def create_error_response(message: str, **kwargs) -> Dict[str, Any]:
    """Create a standardized error response dictionary."""
    response = {"status": "error", "message": message}
    response.update(kwargs)
    return response

def create_success_response(**kwargs) -> Dict[str, Any]:
    """Create a standardized success response dictionary."""
    response = {"status": "success"}
    response.update(kwargs)
    return response

def handle_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    global uploaded_files, retriever, chat_history
    command = cmd.get("command")
    logger.info(f"Handling command: {command}") # Log entry into handle_command
    if command == "upload":
        return handle_upload_command(cmd)
    elif command == "ask":
        return handle_ask_command(cmd)
    elif command == "get_files":
        return {"status": "success", "files": uploaded_files}
    elif command == "get_chat_history":
        return {"status": "success", "chat_history": chat_history}
    elif command == "stop_generation":
        stop_generation_flag[0] = True
        return {"status": "success", "message": "Generation stopped"}
    elif command == "delete_file":
        return handle_delete_file_command(cmd)
    elif command == "reset":
        return handle_reset_command(cmd)
    elif command == "get_status":
        status = {
            "status": "success",
            "num_files": len(uploaded_files),
            "files": uploaded_files,
            "qa_chain_initialized": retriever is not None
        }
        return status
    elif command == "update_file":
        filename = cmd.get("filename")
        if not filename:
            return {"status": "error", "message": "No filename provided"}
        safe_filename = sanitize_filename(filename)
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        if safe_filename not in uploaded_files:
            uploaded_files.append(safe_filename)
        try:
            initialize_qa_system([os.path.join(UPLOAD_FOLDER, f) for f in uploaded_files])
            return {"status": "success", "message": f"File '{safe_filename}' updated", "files": uploaded_files}
        except Exception as e:
            return {"status": "error", "message": f"Failed to update file: {e}"}
    elif command == "get_document_content":
        return handle_get_document_content_command(cmd)
    # elif command == "request_video_generation":
    #     return handle_request_video_generation_command(cmd)
    # elif command == "confirm_video_generation":
    #     return handle_confirm_video_generation_command(cmd)
    # elif command == "generate_video":
    #     return handle_generate_video_command(cmd)
    elif command == "get_supported_file_types":
        return {"status": "success", "supported_file_types": list(ALLOWED_EXTENSIONS)}
    else:
        logger.warning(f"Unknown command received: {command}")
        return {"status": "error", "message": f"Unknown command: {command}"}

def receive_command_data(conn, addr) -> bytes:
    """Receive command data from client connection."""
    buffer = b''
    try:
        while True:
            chunk = conn.recv(1024)
            logger.debug(f"Received chunk of size {len(chunk)} bytes from {addr}")
            if not chunk:
                logger.info(f"Client {addr} disconnected (empty chunk received)")
                return None
            buffer += chunk
            if b'\n' in buffer:
                # Command is complete
                break
        return buffer
    except socket.timeout:
        logger.warning(f"Socket timeout while receiving data from {addr}")
        return None
    except Exception as e:
        logger.error(f"Error receiving data from {addr}: {e}")
        return None

def process_command(buffer: bytes, conn, addr) -> None:
    """Process a complete command from buffer."""
    try:
        cmd = json.loads(buffer.decode('utf-8'))
        logger.info(f"Received command: {cmd.get('command', 'unknown')} from {addr}")
        resp = handle_command(cmd)
        send_response(conn, resp, addr)
    except json.JSONDecodeError:
        # Not a complete JSON yet, continue accumulating
        pass
    except Exception as e:
        logger.error(f"Error processing command: {e}")
        error_resp = {"status": "error", "message": str(e)}
        send_response(conn, error_resp, addr)

def send_response(conn, response: Dict[str, Any], addr) -> None:
    """Send response back to client."""
    try:
        response_data = json.dumps(response).encode('utf-8') + b'\n'
        chunk_size = 1024
        for i in range(0, len(response_data), chunk_size):
            response_chunk = response_data[i:i+chunk_size]
            conn.sendall(response_chunk)
            time.sleep(0.01)  # Small delay between chunks
    except Exception as e:
        logger.error(f"Error sending response to {addr}: {e}")

def handle_client(conn, addr):
    """Handle client connection with improved separation of concerns."""
    logger.info(f"New connection from {addr}")
    try:
        while True:
            buffer = receive_command_data(conn, addr)
            if buffer is None:
                break
            process_command(buffer, conn, addr)
    except ConnectionResetError as e:
        logger.error(f"Connection reset by client {addr}: {e}")
    except BrokenPipeError as e:
        logger.error(f"Broken pipe with client {addr}: {e}")
    except Exception as e:
        logger.error(f"Error handling client {addr}: {e}", exc_info=True)
        try:
            error_resp = {"status": "error", "message": str(e)}
            send_response(conn, error_resp, addr)
        except Exception:
            pass
    finally:
        conn.close()
        logger.info(f"Connection from {addr} closed")

if __name__ == "__main__":
    HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
    PORT = 65432        # Port to listen on (non-privileged ports are > 1023)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Set socket options to reuse address and avoid "Address already in use" errors
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Increase socket buffer size
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)  # 256KB receive buffer
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)  # 256KB send buffer
        
        s.bind((HOST, PORT))
        s.listen(5)  # Allow up to 5 connections in the queue
        logger.info(f"Server is listening on {HOST}:{PORT}")
        try:
            active_threads = []
            while True:
                logger.info("Waiting for new connection...")
                conn, addr = s.accept()
                logger.info(f"Accepted connection from {addr}")
                
                # Set socket timeout for the client connection
                # Increased for long-running operations (e.g., file uploads with embeddings)
                # Adjust based on your hardware/file sizes; 300 seconds = 5 minutes
                conn.settimeout(300)  # Increased from 60 to 300 seconds
                
                client_thread = threading.Thread(target=handle_client, args=(conn, addr))
                client_thread.daemon = True
                client_thread.start()
                
                # Clean up completed threads
                active_threads = [t for t in active_threads if t.is_alive()]
                active_threads.append(client_thread)
                logger.info(f"Active client connections: {len(active_threads)}")
        except KeyboardInterrupt:
            logger.info("Server shutting down...")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            sys.exit(1)