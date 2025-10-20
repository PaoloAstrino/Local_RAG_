import json
import sys
import os
import socket
import requests
import time
import logging
from flask import Flask, request, jsonify, send_from_directory, render_template
from dotenv import load_dotenv

# Get absolute paths
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
template_folder = os.path.join(project_root, 'frontend')
static_folder = os.path.join(project_root, 'frontend', 'static')
UPLOAD_FOLDER = os.path.join(project_root, 'uploads')

app = Flask(__name__, static_folder=static_folder, template_folder=template_folder)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Increase the maximum content length for uploads (e.g., 500MB)
# Adjust this value based on expected total upload size per request
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Download script route ---
@app.route('/download_script/<filename>')
def download_script(filename):
    # Serve the script file from static/scripts for download
    script_dir = os.path.join(app.static_folder, 'scripts')
    return send_from_directory(script_dir, filename, as_attachment=True)

def extract_answer(text):
    parts = text.split("Answer:", 1)
    return parts[1].strip() if len(parts) > 1 else text

def call_external_api(command: dict) -> dict:
    """Attempt to send command via external API."""
    try:
        load_dotenv()
        api_url = os.getenv('API_ENDPOINT_URL')
        api_key = os.getenv('API_KEY')
        if not api_url or not api_key:
            logger.error("API_ENDPOINT_URL or API_KEY not set in .env file.")
            raise ValueError("API_ENDPOINT_URL or API_KEY not set in .env file.")
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {api_key}"
        }
        response = requests.post(api_url, json=command, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"External API call failed with status code {response.status_code}: {response.text}")
            raise Exception(f"External API failed with status {response.status_code}")
    except Exception as e:
        logger.error(f"External API call failed: {e}")
        raise

def send_data_in_chunks(sock, data: bytes, chunk_size: int = 1024) -> None:
    """Send data in chunks to prevent connection issues."""
    total_sent = 0
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        sock.sendall(chunk)
        total_sent += len(chunk)
        logger.debug(f"Sent chunk {i//chunk_size + 1}/{(len(data) + chunk_size - 1)//chunk_size}, total bytes sent: {total_sent}")
        time.sleep(0.01)  # Small delay between chunks
    logger.info(f"Finished sending {total_sent} bytes.")

def receive_response(sock) -> bytes:
    """Receive response from socket until newline or size limit."""
    response_buffer = b""
    while True:
        try:
            part = sock.recv(4096)
            if not part:
                logger.info("Received empty part, server likely closed connection.")
                break
            response_buffer += part
            logger.debug(f"Received part of size {len(part)}, total buffer size: {len(response_buffer)}")
            if response_buffer.endswith(b'\n'):
                 logger.info("Received newline, assuming end of message.")
                 break
            if len(response_buffer) > 10 * 1024 * 1024: # 10MB limit
                logger.warning("Response buffer exceeded 10MB without newline, breaking.")
                break
        except socket.timeout:
            logger.warning("Socket timeout while receiving data.")
            break
        except Exception as recv_e:
            logger.error(f"Error receiving data: {recv_e}")
            break
    logger.info(f"Finished receiving data. Total size: {len(response_buffer)} bytes.")
    return response_buffer

def parse_response(response_buffer: bytes) -> dict:
    """Parse the response buffer into JSON."""
    response_str = response_buffer.decode('utf-8').strip()
    if response_str:
        try:
            if '\n' in response_str:
                response_str = response_str.split('\n')[0]
            parsed_response = json.loads(response_str)
            logger.info("Successfully parsed JSON response.")
            return parsed_response
        except json.JSONDecodeError as json_e:
            logger.error(f"Failed to decode JSON response: {json_e}. Response received: '{response_str[:200]}...'")
            return {"status": "error", "message": f"Failed to decode JSON response from server: {json_e}"}
    else:
         logger.warning("Received empty response string from server.")
         return {"status": "error", "message": "Received empty response from server."}

def communicate_via_socket(command: dict) -> dict:
    """Communicate with local server via socket."""
    try:
        HOST = os.getenv('LOCAL_SERVER_HOST', '127.0.0.1')
        PORT = int(os.getenv('LOCAL_SERVER_PORT', 65432))
        
        logger.info(f"Attempting socket connection to {HOST}:{PORT}...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(60)
            logger.info(f"Connecting to {HOST}:{PORT}...")
            s.connect((HOST, PORT))
            logger.info(f"Connected successfully to {HOST}:{PORT}")
            
            data_to_send = json.dumps(command).encode('utf-8') + b'\n'
            logger.info(f"Preparing to send data of size {len(data_to_send)} bytes...")
            send_data_in_chunks(s, data_to_send)
            
            logger.info("Waiting for response...")
            response_buffer = receive_response(s)
            return parse_response(response_buffer)
            
    except socket.timeout:
        logger.error("Connection to server timed out. Is the server running?")
        return {"status": "error", "message": "Connection to server timed out. Is the server running?"}
    except ConnectionRefusedError:
        logger.error("Could not connect to server. Is the server running?")
        return {"status": "error", "message": "Could not connect to server. Is the server running?"}
    except ConnectionAbortedError:
        logger.error("Connection was aborted. The server may have rejected the request due to size limitations.")
        return {"status": "error", "message": "Connection was aborted. The server may have rejected the request due to size limitations."}
    except BrokenPipeError:
        logger.error("Connection broken. The server may have closed the connection unexpectedly.")
        return {"status": "error", "message": "Connection broken. The server may have closed the connection unexpectedly."}
    except Exception as e:
        logger.error(f"Unexpected error during socket communication: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

def send_command_to_server(command):
    """Send command directly to local server via socket."""
    logger.info(f"Sending command to local server: {command.get('command')}")
    return communicate_via_socket(command)

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/rag_get_chat_history', methods=['GET'])
def get_chat_history():
    command = {"command": "get_chat_history"}
    server_response = send_command_to_server(command)
    # Ensure we have a valid response
    if not isinstance(server_response, dict):
        server_response = {"status": "error", "message": "Invalid response format", "chat_history": []}
    elif "chat_history" not in server_response:
        server_response["chat_history"] = []
    return jsonify(server_response)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/rag_ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '')
    command = {"command": "ask", "question": question}
    server_response = send_command_to_server(command)
    # Ensure we have a valid response even if there's an error
    if not isinstance(server_response, dict):
        server_response = {"status": "error", "message": "Invalid response format", "answer": "Error: Could not process your request"}
    elif "answer" not in server_response:
        # Make sure there's always an answer field for the frontend
        server_response["answer"] = server_response.get("message", "No response received")
    return jsonify(server_response)

@app.route('/rag_upload', methods=['POST'])
def upload_files():
    logger.info("[DEBUG] /rag_upload endpoint called")
    files = request.files.getlist('files')
    logger.info(f"[DEBUG] Number of files received: {len(files)}")
    file_paths = []
    
    # Only save files, do not validate or check extensions/sizes here
    if not os.path.exists(UPLOAD_FOLDER):
        logger.info(f"[DEBUG] Creating upload folder: {UPLOAD_FOLDER}")
        os.makedirs(UPLOAD_FOLDER)
    
    for file in files:
        logger.info(f"[DEBUG] Processing file: {file.filename}")
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        logger.info(f"[DEBUG] Saving file to: {file_path}")
        file.save(file_path)
        file_paths.append(file_path)
    
    if not file_paths:
        logger.info("[DEBUG] No files to upload")
        return jsonify({"status": "error", "message": "No files to upload"})
        
    command = {"command": "upload", "file_paths": file_paths}
    logger.info(f"[DEBUG] Sending upload command to server: {command}")
    server_response = send_command_to_server(command)
    logger.info(f"[DEBUG] Server response: {server_response}")
    return jsonify(server_response)

@app.route('/rag_get_files', methods=['GET'])
def get_files():
    app.logger.info("Received request for /rag_get_files") # Add logging here
    command = {"command": "get_files"}
    server_response = send_command_to_server(command)
    app.logger.info(f"Response from server for get_files: {server_response}") # Add logging here
    # Ensure a valid JSON response even if server communication failed
    if not isinstance(server_response, dict):
         server_response = {"status": "error", "message": "Invalid response from backend server."}
    elif "status" not in server_response:
         server_response["status"] = "unknown" # Add status if missing

    # Ensure 'files' key exists for the frontend JS, even if empty or error
    if "files" not in server_response:
        server_response["files"] = []

    return jsonify(server_response)

@app.route('/rag_stop_generation', methods=['POST'])
def stop_generation():
    command = {"command": "stop_generation"}
    server_response = send_command_to_server(command)
    return jsonify(server_response)

@app.route('/rag_get_status', methods=['GET'])
def get_status():
    command = {"command": "get_status"}
    server_response = send_command_to_server(command)
    return jsonify(server_response)

@app.route('/rag_get_supported_file_types', methods=['GET'])
def get_supported_file_types():
    command = {"command": "get_supported_file_types"}
    server_response = send_command_to_server(command)
    return jsonify(server_response)



@app.route('/rag_request_video_generation', methods=['POST'])
def request_video_generation():
    data = request.json
    command = {
        "command": "request_video_generation",
        "chat_history": data.get("chat_history", [])
    }
    server_response = send_command_to_server(command)
    return jsonify(server_response)

@app.route('/rag_confirm_video_generation', methods=['POST'])
def confirm_video_generation():
    data = request.json
    command = {
        "command": "confirm_video_generation",
        "response": data.get("response"),
        "custom_topic": data.get("custom_topic"),
        "suggested_topic": data.get("suggested_topic"),
        "chat_history": data.get("chat_history", []),
        "script_length": data.get("script_length", "short")  # Add this line
    }
    server_response = send_command_to_server(command)
    return jsonify(server_response)

@app.route('/rag_reset', methods=['POST'])
def reset():
    command = {"command": "reset"}
    server_response = send_command_to_server(command)
    return jsonify(server_response)

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, port=5001, use_reloader=False)  # Disable reloader