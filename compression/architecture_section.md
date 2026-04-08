## 2. System Architecture

### 2.1 Hardware

| Component | Device | Cost |
|---|---|---|
| Preprocessor | Raspberry Pi 5 (4GB RAM) | ~$80 |
| Inference | Jetson Orin Nano Super (8GB) | ~$250 |
| Storage | 2TB NVMe SSD | ~$120 |
| Total | | ~$500 |

The Raspberry Pi 5 runs Raspberry Pi OS Bookworm (ARM64). The Jetson runs JetPack 6.2 with CUDA 12.6. Both devices communicate over a local LAN with no external network access required at inference time.

### 2.2 Pipeline

PDF Document -> PDF Text Extraction (pdfplumber) -> full document text (~5K-65K tokens)
-> Relevance Filter (all-MiniLM-L6-v2) -> top-k chunks (~140-780 tokens)
-> Abbreviation Compressor (domain codebook)
-> HTTP POST to Jetson Orin Nano (llama-server :8080)
-> Qwen2.5-3B-Instruct Q4_K_M via llama.cpp
-> Answer

Stages 0-2 run on the Raspberry Pi 5. Inference runs on the Jetson.

### 2.3 Relevance Filter

Uses sentence-transformers/all-MiniLM-L6-v2 (22M parameters) to embed document chunks and the query into a shared vector space. Cosine similarity selects the top-k most relevant chunks. Documents are chunked into 5-sentence windows. At top-k=5, the filter selects approximately 25 sentences regardless of input document size.

### 2.4 Inference

The Jetson runs llama-server from llama.cpp with CUDA offload at 2,048-token context. Model is Qwen2.5-3B-Instruct Q4_K_M (1.95 GiB). Achieves approximately 19-20 tokens/second. Served via OpenAI-compatible REST API on port 8080.

### 2.5 What Was Not Used

LLMLingua-2 (xlm-roberta-large) was evaluated and disabled. Token-level compression degraded answer coherence on formal government specification prose due to domain mismatch with its meetingbank training data.

Vector database (FAISS, ChromaDB) was not used. For single-document QA, in-memory cosine similarity is sufficient and eliminates a dependency.
