# Nyayora
### *Intelligent Legal Assistant for the New Indian Criminal Law Era*

Nyayora is a state-of-the-art RAG (Retrieval-Augmented Generation) system designed to navigate the complexities of Indian criminal law, providing context-aware analysis across multiple statutes including the IPC, BNS, BNSS, and BSA.

## 🏗️ Technical Architecture & Workflow

The system is divided into two primary phases: the **Build Pipeline** for data preparation and the **Query Pipeline** for real-time inference.

#### 1. Build Pipeline: Multi-Statute Ingestion
```mermaid
graph TD
    A[Source PDFs: IPC, BNS, BNSS, BSA] --> B[PyPDFLoader & Regex Cleaner]
    B --> C[Context-Aware Chunker]
    C --> D{Statute Identification}
    D -->|IPC| E1[IPC Chunks]
    D -->|BNS| E2[BNS Chunks]
    D -->|BNSS| E3[BNSS Chunks]
    D -->|BSA| E4[BSA Chunks]
    
    E1 & E2 & E3 & E4 --> F[Embedding Model: mxbai-embed-large]
    F --> G[(ChromaDB: Separate Collections)]
    
    E1 & E2 & E3 & E4 --> H[BM25 Indexer]
    H --> I[artifacts/chunked_dataset.pkl]
```

#### 2. Query Pipeline: Hybrid RAG Workflow
```mermaid
graph TD
    J[User Query] --> K[Query Classifier Agent]
    K -->|Law Related| L[Intent Classifier Agent]
    K -->|Irrelevant| K_OUT[Generic Response]
    
    L -->|Intent + Target Statutes| M[Multi-Query Expansion]
    M -->|5 Query Variations| N[Hybrid Search Engine]
    
    subgraph Retrieval [Hybrid Retrieval & Fusion]
        N --> O[Semantic Search: ChromaDB Collections]
        N --> P[Sparse Search: BM25 Index]
        O --> Q[Reciprocal Rank Fusion - RRF]
        P --> Q
    end
    
    Q -->|Initial Results| R[FlashRank Reranker]
    R -->|Top 8 Precision Chunks| S[Law Reasoning Agent]
    S --> T[Verification Layer: Citation Cross-Check]
    T --> U[Final Legal Response]
```

### Key Components

1.  **Multi-Database Ingestion**: Unlike standard RAG systems, Nyayora maintains 4 separate ChromaDB collections. This allows the **Intent Classifier** to target specific acts (e.g., searching only BNS and BNSS for modern queries), reducing noise and improving accuracy.
2.  **Hybrid Retrieval**: Combines semantic vector search (`mxbai-embed-large`) with traditional keyword search (BM25) using **Reciprocal Rank Fusion (RRF)**. This ensures that specific legal terms and section numbers are retrieved reliably.
3.  **Two-Stage Ranking**:
    *   **Stage 1**: RRF merges scores from multiple vector and keyword sources.
    *   **Stage 2**: **FlashRank Reranker** processes the top 30 candidates to select the absolute most relevant 8 chunks for the LLM context window.
4.  **Verification Layer**: A post-generation check that validates every cited legal section against the retrieved context to eliminate hallucinations before the user sees the answer.
5.  **Agentic Routing**: Uses specialized mini-agents (`qwen3:0.6b`) for classification and expansion to keep the system fast and responsive on consumer hardware.

## 🚀 Quick Start

1. **Build the Index**:
   ```bash
   python -m src.pipelines.rag_build_pipeline
   ```

2. **Run the Server**:
   ```bash
   streamlit run server.py
   ```
