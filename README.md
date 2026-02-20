# Production-Grade Multi-Tenant RAG System

Intelligent Retrieval • Reranking • Retry Loop • OCR Ingestion • Lifecycle Management

## Overview

This project implements a production-oriented Retrieval-Augmented Generation (RAG) system designed to simulate real-world enterprise AI infrastructure.

The system enables:

Key Features

-  **Multi-tenant document isolation**
-  **OCR-based PDF ingestion**
-  **MMR-based semantic retrieval**
-  **Cross-encoder reranking**
-  **LLM-driven query rewrite retry loop**
-  **Persistent vector storage**
-  **Expiration-aware document lifecycle management**



##  System Flow

```
User Query
     │
     ▼
MMR Retrieval (ChromaDB)
     │
     ▼
Cross-Encoder Reranking
     │
     ▼
LLM Context Judge (YES / NO)
     │
     ├── NO  ──► Rewrite Query ──► Retry Retrieval
     │
     └── YES ──► Generate Grounded Answer
```
## Core Engineering Highlights

### Multi-Tenant Architecture

- **Header-based tenant verification**

- **Metadata-level filtering inside vector DB**

- **Strict isolation of document contexts**

- **Prevents cross-tenant retrieval leakage**

## Intelligent Retrieval Strategy

 ### Max Marginal Relevance (MMR)
 
  - **Reduces redundant chunks**

  - **Increases contextual diversity**

  - **Improves retrieval coverage**

### Cross-Encoder Reranking (MS MARCO MiniLM)

  - **Scores query-document pairs**

  - **Improves semantic precision**

  - **Enhances top-k document quality**

### Hallucination Mitigation Layer

  - **Strict LLM context judge (YES / NO)**

  - **Automatic query rewrite if context is insufficient**

  - **Constrained generation:**
    
     **Answer ONLY from context. If not found, say I don’t know.**

### OCR-Based PDF Ingestion Pipeline

  - **Converts PDFs → Images (pdf2image)**

  - **Extracts text via Tesseract OCR**

  - **Recursive chunk splitting (900 size / 200 overlap)**

  - **Stores metadata:**
    - **filename

    - **page number**

    - **file hash**

    - **tenant_id**

    - **expiration timestamp**

    - **upload timestamp**
   
 ###  Document Lifecycle Management

 - **SHA256-based deduplication**

 - **90-day TTL expiration policy**

 - **Automatic cleanup on server startup**

 - **Persistent ChromaDB storage**

 ### Performance Optimization (ONNX Embeddings)
 
   Includes a custom ONNX embedding implementation:
   - **Loads all-MiniLM-L6-v2 in ONNX format**

   - **Uses ONNX Runtime for CPU-efficient inference**

   - **Implements:**

        - **Tokenization**

        - **Attention mask pooling**

        - **L2 normalization**

  ## Tech Stack

  | Layer           | Technology               |
| --------------- | ------------------------ |
| API Layer       | FastAPI                  |
| Workflow Engine | LangGraph                |
| Vector Database | ChromaDB (Persistent)    |
| LLM             | Ollama (Mistral)         |
| Embeddings      | Ollama / ONNX            |
| Reranker        | Cross-Encoder (MS MARCO) |
| OCR             | Tesseract + pdf2image    |
| Runtime         | Python                   |

## Project Structure

```

├── RAG_pipeline.py         Main FastAPI + LangGraph workflow
├── OnnxEmbeddings.py       Custom ONNX embedding implementation
├── data/upload/            Uploaded PDFs
├── chroma_db/              Persistent vector database

```
     
  
  
