# RAG Pipeline Architecture (Mermaid)

This diagram represents the Retrieval-Augmented Generation (RAG) pipeline as implemented in the MPC RAG Chatbot system, reflecting its hybrid retrieval, augmentation, and generation stages, as well as client-server separation and robust API handling.

```mermaid
flowchart TD
    %% User and Client
    subgraph User_Side[User Side]
        UQ[User Query (Web Interface)]
        C[Client (Flask API)]
    end

    %% Server and RAG Core
    subgraph Server_Side[Server Side]
        direction TB
        SR[RAG Core (server_MPC.py)]
        subgraph Hybrid_Retrieval[Hybrid Retrieval]
            direction LR
            SEM[Semantic Search (Embeddings, Vector Store)]
            BM[BM25 Keyword Search]
            ER[Ensemble Retriever (Weighted Combination)]
            SEM --> ER
            BM --> ER
        end
        AUG[Augmentation (Context, History, Instructions)]
        GEN[Generation (LLM API, Attribution, Fallback)]
    end

    %% Flow
    UQ --> C
    C --> SR
    SR --> ER
    ER --> AUG
    AUG --> GEN
    GEN --> C
    C --> UQ

    %% External LLM
    GEN -- "Prompt & Params" --> LLM[External LLM API (e.g., Gemini)]
    LLM -- "Response" --> GEN

    %% Styling (optional)
    classDef user fill:#f9f,stroke:#333,stroke-width:1px;
    classDef server fill:#bbf,stroke:#333,stroke-width:1px;
    class UQ,C user;
    class SR,ER,AUG,GEN,SEM,BM server;
```

**Legend:**
- User Side: User and client (Flask API)
- Server Side: RAG core, hybrid retrieval, augmentation, generation
- Hybrid retrieval: Combines semantic search (embeddings/vector store) and BM25 keyword search
- Augmentation: Context consolidation, history integration, instruction injection
- Generation: LLM API call, source attribution, fallback handling
- External LLM API: e.g., Google Gemini

You can copy this code into any Mermaid-compatible editor or markdown file to visualize the architecture. Edit node labels or structure as needed to match future changes in your pipeline.
