# Multimodal RAG System

A FastAPI-based RAG system that processes documents using Docling, stores raw files in MinIO, embeds chunks in Milvus, and answers queries using NVIDIA's GPT-OSS-120B with source citations.

## Quick Start

### 1. Start Infrastructure

```bash
docker compose up -d
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your NVIDIA_API_KEY
```

### 3. Install Dependencies

```bash
uv sync
```

### 4. Run the Server

```bash
uv run uvicorn app.main:app --reload --port 8000
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Upload a document
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@your_document.pdf"

# Query with sources
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the document about?", "top_k": 3}'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/documents/upload` | Upload and process document |
| `GET` | `/documents` | List uploaded documents |
| `POST` | `/query` | Query RAG with sources |

## Architecture

- **Docling**: Document processing with `ExportType.DOC_CHUNKS`
- **MinIO**: Raw file storage with presigned URLs
- **Milvus**: Vector storage via `langchain_milvus`
- **NVIDIA GPT-OSS-120B**: LLM via `langchain_nvidia_ai_endpoints`
