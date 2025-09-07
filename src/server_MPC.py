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

# Import video generation functionality
try:
    from heygen_implementation.rag_to_video_generator import rag_conversation_to_video
    VIDEO_GENERATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Video generation not available - {e}")
    VIDEO_GENERATION_AVAILABLE = False
    def rag_conversation_to_video(*args, **kwargs):
        return None, None

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
# --- Original embeddings initialization ---
_underlying_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'} # Specify device (cuda or cpu)
)

# --- Set up cache ---
EMBED_CACHE_DIR = "./embed_cache/"
os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
store = LocalFileStore(EMBED_CACHE_DIR)

# --- Create CacheBackedEmbeddings ---
embeddings = CacheBackedEmbeddings.from_bytes_store(
    _underlying_embeddings, store, namespace=_underlying_embeddings.model_name
)
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

# Modify call_external_llm_api to accept and use chat_history
def call_external_llm_api(question, context, history):
    base_api_url = os.getenv('EXTERNAL_LLM_API_URL') # Should now be just the base URL
    api_key = os.getenv('EXTERNAL_LLM_API_KEY')
    if not base_api_url or not api_key:
        logger.error("LLM API base URL or key not set in environment variables.")
        return "LLM API base URL or key not set in environment variables."

    # Construct the full URL with the API key as a query parameter
    api_url = f"{base_api_url}?key={api_key}"

    prompt_template = get_prompt_template()
    # Format the chat history for the prompt
    formatted_history = format_chat_history(history)
    # Include formatted history in the prompt
    prompt = prompt_template.format(chat_history=formatted_history, context=context, question=question)
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        # Add generationConfig to control output parameters
        "generationConfig": {
            "temperature": 0.6,  # Adjust for creativity vs. factuality (e.g., 0.4-0.7)
            "topP": 0.95,
            "topK": 40,
            "maxOutputTokens": 10024, # Limit the maximum length of the response
            # "stopSequences": ["some_sequence"] # Optional: sequences to stop generation
        }
    }
    headers = {
        "Content-Type": "application/json"
        # No Authorization header
    }
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            try:
                return data['candidates'][0]['content']['parts'][0]['text']
            except Exception:
                return "No answer returned from Gemini API."
        elif response.status_code == 401:
            error_message = (
                "Gemini API authentication failed (401 - UNAUTHENTICATED). "
                "Please verify the following in your Google Cloud Console:\n"
                "1. The API key in your .env file (EXTERNAL_LLM_API_KEY) is correct and active.\n"
                "2. The 'Generative Language API' is enabled for your project.\n"
                "3. Billing is enabled for your project.\n"
                "4. The API key has no restrictions (IP, HTTP referrer) preventing its use from the server.\n"
                "5. The key is correctly appended to the URL (e.g., ...?key=YOUR_KEY)."
            )
            logger.error(error_message)
            return error_message
        else:
            logger.error(f"LLM API error: {response.status_code} {response.text}")
            return f"LLM API error: {response.status_code} {response.text}"
    except Exception as e:
        return f"LLM API call failed: {str(e)}"

def load_documents(file_paths: List[str]) -> List['Document']:
    documents = []
    for file_path in file_paths:
        file_extension = file_path.split('.')[-1].lower()
        try:
            # Sanitize filename before loading
            safe_path = os.path.join(UPLOAD_FOLDER, sanitize_filename(os.path.basename(file_path))) # Use os.path.basename here
            if not os.path.exists(safe_path):
                 logger.warning(f"File not found during loading: {safe_path}")
                 continue # Skip if file doesn't exist at the expected location

            if file_extension == 'pdf':
                loader = PyPDFLoader(safe_path)
                loaded_documents = loader.load()
            elif file_extension == 'csv':
                # Keep CSV loading as is or adapt if needed
                df = pd.read_csv(safe_path)
                # Convert entire CSV to a single document or process rows
                # Simple approach: Convert to string
                content = df.to_string(index=False)
                loaded_documents = [Document(page_content=content, metadata={'source': safe_path})]                # Alternative: JSON approach (as before)
                # json_data = df.to_json(orient='records')
                # loader = JSONLoader(file_path=safe_path, jq_path='.', content_key=None, json_loads=lambda x: json.loads(json_data))
                # loaded_documents = loader.load()
            elif file_extension == 'json':
                 # Keep JSON loading as is
                 with open(safe_path, 'r', encoding='utf-8') as f:
                     try:
                         data = json.load(f)
                     except json.JSONDecodeError:
                         logger.error(f"Invalid JSON file: {safe_path}")
                         continue # Skip invalid JSON

                 # Simplified JSON loading assuming content is text or can be stringified
                 if isinstance(data, list):
                     content = "\n".join([json.dumps(item) for item in data])
                 elif isinstance(data, dict):
                     content = json.dumps(data)
                 else:
                     content = str(data) # Fallback for other JSON structures
                 loaded_documents = [Document(page_content=content, metadata={'source': safe_path})]

                 # Previous JSONLoader approach (keep if preferred)
                 # jq_path = '.[].content' if isinstance(data, list) else '.content'
                 # def text_content_function(record: Dict, metadata: Dict) -> str:
                 #    return record.get("content", "")
                 # loader = JSONLoader(
                 #     file_path=safe_path,
                 #     jq_path=jq_path,
                 #     text_content=True, # Set based on your JSON structure needs
                 #     content_key="content" # Set based on your JSON structure needs
                 # )
                 # loaded_documents = loader.load()
            elif file_extension == 'txt':
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(safe_path, encoding='utf-8')
                loaded_documents = loader.load()
            else:
                # This should not happen if validate_file works correctly, but good as a safeguard
                logger.warning(f"Attempted to load unsupported file type: {file_extension} for file {safe_path}")
                continue # Skip unsupported types explicitly

            # Add metadata (ensure safe_path is used)
            for doc in loaded_documents:
                # Ensure metadata exists and add file_path
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['file_path'] = safe_path # Use the validated, full path
            documents.extend(loaded_documents)
        except Exception as e:
            logger.error(f"Error loading or processing {file_path} (path used: {safe_path}): {e}", exc_info=True) # Log full traceback
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

def handle_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    global uploaded_files, retriever, chat_history
    command = cmd.get("command")
    logger.info(f"Handling command: {command}") # Log entry into handle_command
    if command == "upload":
        file_paths = cmd.get("file_paths", [])
        if not file_paths:
            logger.warning("Upload command received with no file paths.")
            return {"status": "error", "message": "No files provided"}

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
             return {"status": "error", "message": f"Validation failed for files: {', '.join(invalid_files_info)}"}

        # Proceed only with valid file paths corresponding to the validated names
        valid_file_paths = [os.path.join(UPLOAD_FOLDER, fname) for fname in valid_files_info]
        file_names = valid_files_info # Use the validated names

        logger.info(f"Attempting to initialize QA system with {len(valid_file_paths)} valid files...")
        success = initialize_qa_system(valid_file_paths) # Pass full paths

        if success:
            uploaded_files = file_names # Store only basenames
            logger.info(f"Upload successful. Current files: {uploaded_files}")
            return {
                "status": "success",
                "message": "Files uploaded and retriever initialized successfully",
                "files": file_names
            }
        else:
            # Initialization failed, error already logged in initialize_qa_system
            logger.error("Upload command failed due to QA system initialization error.")
            return {"status": "error", "message": "Failed to initialize retriever with uploaded files. Check server logs for details."}
    elif command == "ask":
        query = cmd.get("question", "")
        if retriever is None:
            return {"status": "error", "message": "Retriever not initialized. Upload files first.", "answer": ""}
        try:
            logger.info(f"Retrieving documents for query: {query}")
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
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
    elif command == "get_files":
        return {"status": "success", "files": uploaded_files}
    elif command == "get_chat_history":
        return {"status": "success", "chat_history": chat_history}
    elif command == "stop_generation":
        stop_generation_flag[0] = True
        return {"status": "success", "message": "Generation stopped"}
    elif command == "delete_file":
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
    elif command == "reset":
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
                # Alternative: Read with pandas and convert to string
                # df = pd.read_csv(file_path)
                # content = df.to_string(index=False)            elif file_extension == "json":
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
    elif command == "request_video_generation":
        # This command now requires the chat history to be passed in.
        client_chat_history = cmd.get("chat_history", [])
        if not client_chat_history:
            return {"status": "error", "message": "No conversation history available for video generation"}
        
        try:
            # Analyze the conversation to identify the main topic
            conversation_summary = ""
            # Use the client-provided chat history
            recent_messages = client_chat_history[-3:] if len(client_chat_history) >= 3 else client_chat_history
            
            for i, entry in enumerate(recent_messages, 1):
                # Entry can be [question, answer] or [question, answer, docs]
                question, answer = entry[0], entry[1]
                conversation_summary += f"Q{i}: {question[:100]}...\nA{i}: {answer[:150]}...\n\n"
            
            # Use LLM to identify the main topic for confirmation
            topic_analysis_prompt = f"""
Based on this recent conversation, identify the main topic or subject being discussed. 
Provide a clear, concise topic name (2-8 words) that captures what the conversation is about.

Recent conversation:
{conversation_summary}

Respond with just the topic name, nothing else:
"""
            
            main_topic = call_external_llm_api(topic_analysis_prompt, "", [])
            main_topic = main_topic.strip()
            
            # Ask user for confirmation
            confirmation_message = f"Before starting video generation, can you confirm that the main topic you'd like the video to cover is: '{main_topic}'? Please respond 'yes' to proceed or provide the specific topic you'd prefer."
            
            return {
                "status": "success",
                "message": "video_confirmation_requested",
                "confirmation_question": confirmation_message,
                "suggested_topic": main_topic,
                "conversation_length": len(client_chat_history)
            }
            
        except Exception as e:
            logger.error(f"Error during video generation request: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to analyze conversation: {str(e)}"}
    
    elif command == "confirm_video_generation":
        # This command also requires the chat history to be passed in.
        user_response = cmd.get("response", "").strip().lower()
        custom_topic = cmd.get("custom_topic", "").strip()
        client_chat_history = cmd.get("chat_history", [])
        script_length = cmd.get("script_length", "short") # Default to 'short'
        
        if not client_chat_history:
            return {"status": "error", "message": "No conversation history was provided for video generation."}
        
        if not VIDEO_GENERATION_AVAILABLE:
            return {"status": "error", "message": "Video generation functionality not available. Check if heygen_implementation module is properly installed."}
        
        # Determine if user confirmed or wants to proceed
        confirmed = user_response in ["yes", "y", "confirm", "proceed", "ok", "sure"]
        
        if not confirmed and not custom_topic:
            return {"status": "error", "message": "Video generation cancelled. Please confirm the topic or provide a specific topic."}
        
        try:
            # Use custom topic if provided, otherwise use the conversation
            final_topic = custom_topic if custom_topic else cmd.get("suggested_topic", "Conversation Summary")
            
            # Build comprehensive conversation context with retrieved documents
            full_conversation = ""
            all_questions = []
            all_answers = []
            all_retrieved_docs = []
            
            # Use the client-provided chat history
            for i, entry in enumerate(client_chat_history, 1):
                if len(entry) >= 2:
                    question, answer = entry[0], entry[1]
                    retrieved_docs = entry[2] if len(entry) > 2 else []
                    
                    full_conversation += f"Exchange {i}:\nQ: {question}\nA: {answer}\n\n"
                    all_questions.append(question)
                    all_answers.append(answer)
                    all_retrieved_docs.extend(retrieved_docs)
            
            # Create retrieved documents section
            retrieved_docs_context = ""
            if all_retrieved_docs:
                retrieved_docs_context = "\n\nRETRIEVED DOCUMENT CHUNKS (PRIMARY SOURCE MATERIAL):\n"
                for i, doc in enumerate(all_retrieved_docs, 1):
                    retrieved_docs_context += f"Document Chunk {i}: {doc}\n\n"
            
            # Create a comprehensive context for video generation
            conversation_context = f"""
            VIDEO TOPIC: {final_topic}
            
            FULL CONVERSATION HISTORY ({len(client_chat_history)} exchanges):
            {full_conversation}
            
            MAIN QUESTIONS DISCUSSED:
            {chr(10).join(f'- {q}' for q in all_questions)}
            
            {retrieved_docs_context}
            """
            
            logger.info(f"Generating script for topic: {final_topic} with {len(client_chat_history)} conversation exchanges and {len(all_retrieved_docs)} retrieved document chunks...")
            
            script_result = rag_conversation_to_video(
                conversation_context=conversation_context,
                title=f"{final_topic} - RAG Conversation Summary",
                script_length=script_length
            )

            # Check for refusal phrases from the LLM
            if not script_result or not script_result.get("script"):
                logger.error(f"LLM refused to generate script or script generation failed.")
                return {"status": "error", "message": "I am sorry, I am unable to generate a script based on the current context."}

            logger.info("Script generation successful")
            return {
                "status": "success",
                "message": "Script generated successfully from conversation history",
                "script": script_result,
                "title": f"{final_topic} - RAG Conversation Summary",
                "conversation_exchanges": len(client_chat_history),
                "topic": final_topic
            }
                
        except Exception as e:
            logger.error(f"Error during video generation: {e}", exc_info=True)
            return {"status": "error", "message": f"Video generation error: {str(e)}"}
            
    elif command == "generate_video":
        # Check if video generation is available
        if not VIDEO_GENERATION_AVAILABLE:
            return {"status": "error", "message": "Video generation functionality not available. Check if heygen_implementation module is properly installed."}
            
        # Generate video from RAG conversation context
        question = cmd.get("question", "")
        answer = cmd.get("answer", "")
        retrieved_docs = cmd.get("retrieved_docs", [])
        title = cmd.get("title", "RAG Response Video")
        
        if not question or not answer:
            return {"status": "error", "message": "Question and answer are required for video generation"}

        # Only enforce minimum limits for video generation
        if len(question.strip()) < 10:
            return {"status": "error", "message": "Question too short for video generation (minimum 10 characters)"}

        if len(answer.strip()) < 50:
            return {"status": "error", "message": "Answer too short for video generation (minimum 50 characters)"}

        if not retrieved_docs or len(retrieved_docs) == 0:
            return {"status": "error", "message": "No retrieved documents available for video generation"}

        # Check if retrieved documents have meaningful content
        total_doc_length = sum(len(str(doc).strip()) for doc in retrieved_docs)
        if total_doc_length < 100:
            return {"status": "error", "message": "Insufficient document content for video generation (minimum 100 characters)"}

        logger.info(f"Video generation validation passed - Question: {len(question)} chars, Answer: {len(answer)} chars, Docs: {len(retrieved_docs)} items, Total doc content: {total_doc_length} chars")
        
        try:
            # Format the conversation context for script generation
            conversation_context = f"""
QUESTION: {question}

RETRIEVED DOCUMENT CHUNKS (PRIMARY SOURCE MATERIAL):
"""
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs, 1):
                    conversation_context += f"Document Chunk {i}: {doc}\n\n"
            else:
                conversation_context += "- No specific documents retrieved\n"
            conversation_context += f"""

RAG SYSTEM RESPONSE:
{answer}
"""
            logger.info(f"Generating script for question: {question[:50]}...")
            # Call external script generation function
            script_text = rag_conversation_to_video(
                conversation_context=conversation_context,
                title=title
            )
            if script_text:
                logger.info("Script generation successful")
                return {
                    "status": "success",
                    "message": "Script generated successfully",
                    "script": script_text,
                    "title": title
                }
            else:
                logger.error("Script generation failed")
                return {"status": "error", "message": "Failed to generate script"}
                
        except Exception as e:
            logger.error(f"Error during video generation: {e}", exc_info=True)
            return {"status": "error", "message": f"Video generation error: {str(e)}"}
    elif command == "get_supported_file_types":
        return {"status": "success", "supported_file_types": list(ALLOWED_EXTENSIONS)}
    else:
        logger.warning(f"Unknown command received: {command}")
        return {"status": "error", "message": f"Unknown command: {command}"}

def handle_client(conn, addr):
    logger.info(f"New connection from {addr}")
    buffer = b''
    try:
        while True:
            try:
                # Increase receive timeout if processing might take long
                # conn.settimeout(60) # Example: 60 seconds
                chunk = conn.recv(1024)
                logger.debug(f"Received chunk of size {len(chunk)} bytes from {addr}")
                if not chunk:
                    logger.info(f"Client {addr} disconnected (empty chunk received)")
                    break
                buffer += chunk
                try:
                    cmd = json.loads(buffer.decode('utf-8'))
                    logger.info(f"Received command: {cmd.get('command', 'unknown')} from {addr}")
                    resp = handle_command(cmd)
                    logger.debug(f"Sending response of size {len(json.dumps(resp))} bytes to {addr}")
                    response_data = json.dumps(resp).encode('utf-8') + b'\n'
                    chunk_size = 1024
                    for i in range(0, len(response_data), chunk_size):
                        response_chunk = response_data[i:i+chunk_size]
                        conn.sendall(response_chunk)
                        time.sleep(0.01)  # Small delay between chunks
                    buffer = b''
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing command: {e}")
                    error_resp = {"status": "error", "message": str(e)}
                    try:
                        conn.sendall(json.dumps(error_resp).encode('utf-8') + b'\n')
                    except Exception:
                        pass
                    buffer = b''  # Clear buffer after error
            except socket.timeout:
                logger.warning(f"Socket timeout while receiving data from {addr}")
                break
    except ConnectionResetError as e:
        logger.error(f"Connection reset by client {addr}: {e}")
    except BrokenPipeError as e:
        logger.error(f"Broken pipe with client {addr}: {e}")
    except Exception as e:
        logger.error(f"Error handling client {addr}: {e}", exc_info=True)
        try:
            error_resp = {"status": "error", "message": str(e)}
            conn.sendall(json.dumps(error_resp).encode('utf-8') + b'\n')
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
                
                # Set socket timeout for the client connection (e.g., 60 seconds)
                # Increase if server processing (like file loading) might take long
                conn.settimeout(60)  # 60 second timeout
                
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