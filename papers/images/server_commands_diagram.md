<!--
Main interaction flow for the RAG chatbot system:
- Only the main command (ask) is directly connected to the Client.
- All other commands are shown as arrows branching from the main command (ask), visually emphasizing ask as the central hub.
-->

```mermaid
flowchart LR
    U[User] --> C[Client]
    C --> ask["ask"]

    %% All other commands branch from ask
    ask --> get_chat_history["get_chat_history"]
    ask --> get_files["get_files"]
    ask --> upload["upload"]
    ask --> update_file["update_file"]
    ask --> get_supported_file_types["get_supported_file_types"]
    ask --> get_document_content["get_document_content"]
    ask --> stop_generation["stop_generation"]
    ask --> delete_file["delete_file"]
    ask --> reset["reset"]
    ask --> get_status["get_status"]

    %% Color classes
    classDef user fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1;
    classDef client fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20;
    classDef server fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#4a148c;

    class U user;
    class C client;
    class ask,get_chat_history,get_files,upload,stop_generation,delete_file,reset,update_file,get_document_content,get_supported_file_types,get_status server;
```
