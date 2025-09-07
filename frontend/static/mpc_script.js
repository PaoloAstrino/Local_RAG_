// MPC Chatbot Client Script
// Handles file upload, chat, stop generation, and download chat history for app_MPC.py

let currentRequest = null;

function showMessage(sender, message) {
  const chatArea = document.getElementById("chat-area");
  const senderClass =
    sender === "You"
      ? "you-message"
      : sender === "Assistant"
      ? "assistant-message"
      : "system-message";
  chatArea.innerHTML += `<div class="message ${senderClass}"><strong>${sender}:</strong> ${message}</div>`;
  chatArea.scrollTop = chatArea.scrollHeight;
}

function updateFileList(fileNames) {
  let fileListContainer = document.getElementById("file-list-container");
  if (!fileListContainer) {
    fileListContainer = document.createElement("div");
    fileListContainer.id = "file-list-container";
    fileListContainer.className = "file-list-container";
    fileListContainer.innerHTML =
      '<h3>Uploaded Documents:</h3><ul id="file-list"></ul>';
    document.querySelector(".upload-area").appendChild(fileListContainer);
  }
  const fileList = document.getElementById("file-list");
  fileList.innerHTML = "";
  fileNames.forEach((fileName) => {
    const listItem = document.createElement("li");
    listItem.textContent = fileName;
    fileList.appendChild(listItem);
  });
}

// Function to update the displayed file list (assuming it exists)
// If you have a different function name or mechanism, adapt this part.
function updateFileListUI(files) {
  const fileListElement = document.getElementById("file-list"); // Assuming you have an element with id="file-list"
  if (fileListElement) {
    fileListElement.innerHTML = ""; // Clear existing list
    if (files && files.length > 0) {
      const list = document.createElement("ul");
      files.forEach((file) => {
        const listItem = document.createElement("li");
        listItem.textContent = file;
        list.appendChild(listItem);
      });
      fileListElement.appendChild(list);
    } else {
      fileListElement.textContent = "No documents uploaded.";
    }
  }
  // Also update the file selection span if needed
  const fileSelectionSpan = document.getElementById("file-selection");
  if (fileSelectionSpan) {
    fileSelectionSpan.textContent =
      files && files.length > 0
        ? `${files.length} file(s) loaded`
        : "No file chosen";
  }
}

// Function to fetch and update the file list from the server
async function fetchAndUpdateFileList() {
  try {
    const response = await fetch("/mpc_get_files");
    const data = await response.json();
    if (data.status === "success" && data.files) {
      updateFileListUI(data.files);
    } else {
      console.error("Failed to fetch file list:", data.message);
      updateFileListUI([]); // Clear list on error
    }
  } catch (error) {
    console.error("Error fetching file list:", error);
    updateFileListUI([]); // Clear list on error
  }
}

// Call this on page load to initialize the file list
document.addEventListener("DOMContentLoaded", fetchAndUpdateFileList);

function uploadFiles() {
  const fileInput = document.getElementById("file-upload");
  const files = fileInput.files;
  if (files.length === 0) {
    showMessage("System", "No files selected. Please choose files to upload.");
    return;
  }
  let fileNames = Array.from(files)
    .map((file) => file.name)
    .join(", ");
  showMessage("System", `Loading documents: ${fileNames}...`);
  const loadingIndicator = document.createElement("div");
  loadingIndicator.id = "loading-indicator";
  loadingIndicator.innerHTML =
    '<div class="spinner"></div><p>Processing documents...</p>';
  document.querySelector(".chat-container").appendChild(loadingIndicator);
  // Use FormData to send files to a local backend or Electron bridge
  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append("files", files[i]);
  }
  // For local protocol, use fetch to /upload if running a local server, or use a custom bridge
  fetch("/mpc_upload", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("loading-indicator").remove();
      if (data.status === "success") {
        showMessage("System", `Files uploaded successfully: ${fileNames}`);
        showMessage(
          "System",
          "You can now start asking questions about these documents."
        );
        if (data.files && data.files.length > 0) {
          updateFileList(data.files);
        }
      } else {
        showMessage("System", `${data.message}`);
      }
    })
    .catch((error) => {
      if (document.getElementById("loading-indicator")) {
        document.getElementById("loading-indicator").remove();
      }
      showMessage("System", "Error uploading files. Please try again.");
    });
  fileInput.value = "";
}

document.addEventListener("DOMContentLoaded", function () {
  const userInput = document.getElementById("user-input");
  userInput.addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
      event.preventDefault();
      sendMessage();
    }
  });
  const fileInput = document.getElementById("file-upload");
  fileInput.addEventListener("change", function () {
    const fileSelection = document.getElementById("file-selection");
    if (fileInput.files.length > 0) {
      if (fileInput.files.length === 1) {
        fileSelection.textContent = fileInput.files[0].name;
      } else {
        fileSelection.textContent = `${fileInput.files.length} files selected`;
      }
    } else {
      fileSelection.textContent = "No file chosen";
    }
  });
  fetch("/mpc_get_files")
    .then((response) => response.json())
    .then((data) => {
      if (data.files && data.files.length > 0) {
        updateFileList(data.files);
      }
    })
    .catch((error) => {});
});

function sendMessage() {
  const userInput = document.getElementById("user-input").value;
  if (userInput.trim() === "") return;
  showMessage("You", userInput);
  const typingIndicator = document.createElement("div");
  typingIndicator.id = "typing-indicator";
  typingIndicator.innerHTML =
    '<div class="typing-dots"><span></span><span></span><span></span></div>';
  document.getElementById("chat-area").appendChild(typingIndicator);
  const stopButton = document.getElementById("stop-generation");
  stopButton.disabled = false;
  const controller = new AbortController();
  currentRequest = controller;
  fetch("/mpc_ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: userInput }),
    signal: controller.signal,
  })
    .then((response) => response.json())
    .then((data) => {
      if (document.getElementById("typing-indicator")) {
        document.getElementById("typing-indicator").remove();
      }
      showMessage("Assistant", data.answer);
      stopButton.disabled = true;
      currentRequest = null;
    })
    .catch((error) => {
      if (document.getElementById("typing-indicator")) {
        document.getElementById("typing-indicator").remove();
      }
      if (error.name !== "AbortError") {
        showMessage(
          "System",
          "Error processing your question. Please try again."
        );
      }
      stopButton.disabled = true;
      currentRequest = null;
    });
  document.getElementById("user-input").value = "";
}

function stopGeneration() {
  if (currentRequest) {
    currentRequest.abort();
    currentRequest = null;
    fetch("/mpc_stop_generation", {
      method: "POST",
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.status === "success") {
          showMessage("System", "Response generation stopped.");
        }
      })
      .catch((error) => {});
    if (document.getElementById("typing-indicator")) {
      document.getElementById("typing-indicator").remove();
    }
    document.getElementById("stop-generation").disabled = true;
  }
}

function downloadChat() {
  const downloadButton = document.getElementById("download-chat");
  const originalText = downloadButton.textContent;
  downloadButton.textContent = "Preparing...";
  downloadButton.disabled = true;
  fetch("/mpc_get_chat_history")
    .then((response) => response.json())
    .then((data) => {
      const currentDate = new Date();
      const formattedDate = currentDate.toLocaleString();
      let chatContent = "Chat History - " + formattedDate + "\n";
      chatContent += "MPC Project - Document Chatbot\n\n";
      if (data.chat_history && data.chat_history.length > 0) {
        data.chat_history.forEach((item, index) => {
          const question = item[0];
          const answer = item[1];
          const timestamp = new Date(
            currentDate - (data.chat_history.length - index) * 60000
          ).toLocaleTimeString();
          chatContent += `[${timestamp}] You: ${question}\n\n`;
          chatContent += `[${timestamp}] Assistant: ${answer}\n\n`;
        });
      } else {
        const chatArea = document.getElementById("chat-area");
        const messages = chatArea.getElementsByClassName("message");
        if (messages.length === 0) {
          showMessage("System", "No messages to download.");
          downloadButton.textContent = originalText;
          downloadButton.disabled = false;
          return;
        }
        for (let i = 0; i < messages.length; i++) {
          const message = messages[i];
          const sender = message.classList.contains("you-message")
            ? "You"
            : message.classList.contains("assistant-message")
            ? "Assistant"
            : "System";
          let messageText = message.textContent.trim();
          if (messageText.startsWith(sender + ":")) {
            messageText = messageText.substring(sender.length + 1).trim();
          }
          const timestamp = new Date(
            currentDate - (messages.length - i) * 60000
          ).toLocaleTimeString();
          chatContent += `[${timestamp}] ${sender}: ${messageText}\n\n`;
        }
      }
      const blob = new Blob([chatContent], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download =
        "mpc_chat_history_" + currentDate.toISOString().slice(0, 10) + ".txt";
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        downloadButton.textContent = originalText;
        downloadButton.disabled = false;
      }, 100);
    })
    .catch((error) => {
      downloadButton.textContent = originalText;
      downloadButton.disabled = false;
      showMessage(
        "System",
        "Error downloading chat history. Please try again."
      );
    });
}

async function resetSystem() {
  displayMessage("Resetting system...", "system");
  try {
    const response = await fetch("/mpc_reset", {
      method: "POST",
    });
    const data = await response.json();
    if (data.status === "success") {
      displayMessage("System reset successfully.", "system");
      // Clear the chat history display
      document.getElementById("chat-area").innerHTML = "";
      // Clear the file list display on the frontend
      updateFileListUI([]); // <--- Add this line
    } else {
      displayMessage(`Error resetting system: ${data.message}`, "error");
    }
  } catch (error) {
    displayMessage(`Error resetting system: ${error}`, "error");
  }
}

function getStatus() {
  fetch("/mpc_get_status")
    .then((res) => res.json())
    .then((data) => showMessage("System", JSON.stringify(data, null, 2)));
}

function getSupportedFileTypes() {
  fetch("/mpc_get_supported_file_types")
    .then((res) => res.json())
    .then((data) =>
      showMessage(
        "System",
        "Supported types: " + (data.supported_file_types || []).join(", ")
      )
    );
}

async function deleteFile() {
  const filename = document.getElementById("filename-action").value;
  if (!filename) {
    displayMessage("Please enter a filename to delete.", "error");
    return;
  }
  displayMessage(`Attempting to delete ${filename}...`, "system");
  try {
    const response = await fetch("/mpc_delete_file", {
      // Assuming endpoint exists
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: filename }),
    });
    const data = await response.json();
    if (data.status === "success") {
      displayMessage(data.message || `File ${filename} deleted.`, "system");
      updateFileListUI(data.files || []); // Update UI
    } else {
      displayMessage(`Error deleting file: ${data.message}`, "error");
    }
  } catch (error) {
    displayMessage(`Error deleting file: ${error}`, "error");
  }
}

async function updateFile() {
  const filename = document.getElementById("filename-action").value;
  if (!filename) {
    displayMessage("Please enter a filename to update.", "error");
    return;
  }
  displayMessage(`Attempting to update/re-index ${filename}...`, "system");
  try {
    const response = await fetch("/mpc_update_file", {
      // Assuming endpoint exists
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: filename }),
    });
    const data = await response.json();
    if (data.status === "success") {
      displayMessage(
        data.message || `File ${filename} updated/re-indexed.`,
        "system"
      );
      updateFileListUI(data.files || []); // Update UI
    } else {
      displayMessage(`Error updating file: ${data.message}`, "error");
    }
  } catch (error) {
    displayMessage(`Error updating file: ${error}`, "error");
  }
}

function getDocumentContent() {
  const filename = document.getElementById("filename-action").value;
  if (!filename) return showMessage("System", "Enter a filename.");
  fetch("/mpc_get_document_content", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  })
    .then((res) => res.json())
    .then((data) => {
      if (data.status === "success") {
        showMessage(
          "System",
          `Content of ${data.filename}:<br><pre>${data.content}</pre>`
        );
      } else {
        showMessage("System", data.message || JSON.stringify(data));
      }
    });
}

// --- Video Generation Functions ---

// Helper function to get chat history from the DOM
function getChatHistoryFromDOM() {
    console.log("getChatHistoryFromDOM called");
    const chatArea = document.getElementById("chat-area");
    const messages = chatArea.getElementsByClassName("message");
    const history = [];
    let currentQuestion = null;
    console.log(`Found ${messages.length} messages.`);

    for (let i = 0; i < messages.length; i++) {
        const message = messages[i];
        const isYou = message.classList.contains("you-message");
        const isAssistant = message.classList.contains("assistant-message");
        
        const strongTag = message.querySelector('strong');
        if (!strongTag) {
            console.warn("Message element without strong tag found, skipping:", message);
            continue; // Skip if message format is unexpected
        }

        const textContent = message.textContent.replace(strongTag.innerText, '').trim();
        console.log(`Processing message ${i}: Sender: ${strongTag.innerText}, Content: '${textContent}'`);

        if (isYou) {
            currentQuestion = textContent;
            console.log(`Stored question: ${currentQuestion}`);
        } else if (isAssistant && currentQuestion) {
            history.push([currentQuestion, textContent, []]);
            console.log(`Pushed to history: ["${currentQuestion}", "${textContent}"]`);
            currentQuestion = null; // Reset for the next Q&A pair
        }
    }
    console.log("Final history object:", history);
    return history;
}

// 1. Request video generation (topic analysis)
async function requestVideoGeneration() {
    const scriptButton = document.getElementById("generate-script");
    scriptButton.disabled = true;
    scriptButton.textContent = "Analyzing...";

    const chatHistory = getChatHistoryFromDOM();

    if (chatHistory.length === 0) {
        showMessage("System", "No conversation history to generate a script from.");
        scriptButton.disabled = false;
        scriptButton.textContent = "Generate Script";
        return;
    }

    try {
        const response = await fetch("/mpc_request_video_generation", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ chat_history: chatHistory }),
        });
        const data = await response.json();

        if (data.status === "success" && data.message === "video_confirmation_requested") {
            showVideoConfirmationDialog(data.confirmation_question, data.suggested_topic, chatHistory);
        } else {
            showMessage("System", data.message || "Could not start video generation process.");
        }
    } catch (error) {
        showMessage("System", "Error requesting video generation: " + error);
    } finally {
        scriptButton.disabled = false;
        scriptButton.textContent = "Generate Script";
    }
}

// 2. Show confirmation dialog
function showVideoConfirmationDialog(confirmationQuestion, suggestedTopic, chatHistory) {
    const modal = document.createElement("div");
    modal.id = "script-confirmation-modal";
    modal.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-color: rgba(0, 0, 0, 0.7); display: flex;
        justify-content: center; align-items: center; z-index: 1000;
    `;

    const modalContent = document.createElement("div");
    modalContent.style.cssText = `
        background-color: #2a2a2a; color: #f5f5f5; padding: 20px;
        border-radius: 8px; max-width: 500px; text-align: center; border: 1px solid #444;
    `;

    modalContent.innerHTML = `
        <h3>Script Generation Confirmation</h3>
        <p>${confirmationQuestion}</p>
        <input type="text" id="custom-topic-input" placeholder="Or enter a different topic here" style="width: 80%; padding: 8px; margin-top: 10px;"/>
        <div style="margin-top: 15px; text-align: left; margin-left: 10%;">
            <label style="display: block; margin-bottom: 5px;">Script Length:</label>
            <input type="radio" id="script-length-short" name="script-length" value="short" checked>
            <label for="script-length-short" style="margin-right: 15px;">Short (1 minute)</label>
            <input type="radio" id="script-length-medium" name="script-length" value="medium">
            <label for="script-length-medium">Medium (5 minutes)</label>
        </div>
        <div style="margin-top: 20px;">
            <button id="confirm-yes-btn" style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; margin-right: 10px; cursor: pointer;">
                Yes, Proceed
            </button>
            <button onclick="cancelVideoGeneration()" style="background-color: #f44336; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;">
                Cancel
            </button>
        </div>
    `;

    modal.appendChild(modalContent);
    document.body.appendChild(modal);

    document.getElementById("confirm-yes-btn").onclick = () => {
        const customTopic = document.getElementById("custom-topic-input").value;
        const selectedLength = document.querySelector('input[name="script-length"]:checked').value;
        confirmVideoGeneration(suggestedTopic, customTopic, chatHistory, selectedLength);
    };
}

// 3. Confirm and start generation
async function confirmVideoGeneration(suggestedTopic, customTopic, chatHistory, selectedLength) {
    const modal = document.getElementById("script-confirmation-modal");
    if (modal) modal.remove();

    showMessage("System", "Script generation confirmed. Starting process...");
    showMessage("System", "Generating script from conversation history..."); // Added message
    const progressDiv = document.createElement("div");
    progressDiv.id = "script-progress";
    progressDiv.innerHTML = "<p>Generating script... (This may take a moment)</p>";
    document.getElementById("chat-area").appendChild(progressDiv);

    try {
        const response = await fetch("/mpc_confirm_video_generation", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                response: "yes",
                custom_topic: customTopic,
                suggested_topic: suggestedTopic,
                chat_history: chatHistory,
                script_length: selectedLength,
            }),
        });
        const data = await response.json();

        if (progressDiv) progressDiv.remove();

        if (data.status === "success" && data.script) {
            const fullScript = data.script.script;
            const maxLength = 500; // Maximum characters to display
            let displayedScript = fullScript;
            let downloadLinkHtml = '';

            if (fullScript.length > maxLength) {
                displayedScript = fullScript.substring(0, maxLength) + '...';
            }

            let scriptHtml = `<strong>Script for topic: ${data.topic}</strong><br><pre>${displayedScript}</pre>`;
            
            if (data.script.filename) {
                downloadLinkHtml = `<br><a class="download-script-link" href="/download_script/${data.script.filename}" download>Download Full Script</a>`;
            }
            
            showMessage("System", scriptHtml + downloadLinkHtml);
        } else {
            showMessage("System", data.message || "Failed to generate script.");
        }
    } catch (error) {
        if (progressDiv) progressDiv.remove();
        showMessage("System", "Error during script generation: " + error);
    }
}

function cancelVideoGeneration() {
    const modal = document.getElementById("script-confirmation-modal");
    if (modal) modal.remove();
    showMessage("System", "Script generation cancelled.");
}


// This function is now a wrapper for the new workflow
function generateScript() {
    requestVideoGeneration();
}

// Add CSS for the spinning animation
const style = document.createElement("style");
style.textContent = `
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);

// Helper function to display messages with different types
function displayMessage(message, type) {
  const messageClass =
    type === "error"
      ? "system-message"
      : type === "system"
      ? "system-message"
      : "assistant-message";
  showMessage(type === "error" ? "Error" : "System", message);
}
