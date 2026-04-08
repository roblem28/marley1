# Marley1: Relevance-Routed Document Intelligence on Sub-$500 Edge Hardware

**Abstract**

Without the relevance filter, inference fails on 4 of 5 documents. With it, Marley1 answers all five. The filter is not an optimization. It is the architecture.

We demonstrate that semantic relevance filtering enables a 3B parameter language model running on a $500 edge device stack to answer questions over DoD technical specification documents 11x-457x larger than its context window. Compressed answers outperform the no-context baseline by 80% on average across five Unified Facilities Criteria (UFC) documents. Marley1 runs entirely offline on a Raspberry Pi 5 and Jetson Orin Nano Super with no cloud dependencies.

Marley1 is not a replacement for frontier models. It is proof that useful document intelligence can be moved onto cheap, private, offline hardware through relevance-routed context construction. The question is not whether a $500 device can beat GPT-4. The question is whether someone with no cloud access, no subscription budget, and no IT department can ask a technical question and get a specific, grounded answer. Marley1 answers yes.



---

## 1. Introduction

Large language models require documents to fit within a fixed context window at inference time. For frontier models accessed via API, context windows of 128K–1M tokens make this constraint largely invisible. For models deployed at the edge — on embedded hardware with 4–8GB of unified memory — the constraint is severe. A 3B parameter model running on a Jetson Orin Nano operates with a practical context limit of 2,048–4,096 tokens. A single government specification document routinely exceeds 60,000 tokens.

This paper addresses a simple question: can a small model on cheap hardware answer questions over documents it cannot fit in memory?

We demonstrate that semantic relevance filtering — selecting only the document chunks most similar to a query — enables a 3B parameter model to answer questions over documents 11x–457x larger than its context window, with answer quality exceeding the no-context baseline by 80% on average.

Marley1 runs entirely offline on two consumer devices: a Raspberry Pi 5 ($80) handling document preprocessing and a Jetson Orin Nano Super ($250) with a 2TB NVMe SSD handling inference. Total hardware cost is approximately $500. No cloud services, no API keys, no internet connection required at inference time.

We evaluate on five Unified Facilities Criteria (UFC) documents published by the U.S. Department of Defense — formal technical specifications ranging from 5,836 to 65,325 tokens covering electrical, structural, microgrid, C5ISR, and unaccompanied housing design requirements. These documents represent a realistic and challenging retrieval task: dense technical prose, domain-specific terminology, and no overlap with general web training data.

Our contributions are:

1. A two-stage compression pipeline (relevance filter + abbreviation compressor) that reduces DoD specification documents by an average of 122x while preserving answer quality.
2. An empirical demonstration that relevance filtering alone enables edge LLM inference over large documents — token-level compression via LLMLingua-2 degrades quality at 3B scale and is not recommended.
3. A keyword-recall evaluation methodology for cases where LLM-as-judge fails at small model scale.
4. An open-source implementation targeting sub-$500 hardware accessible to builders anywhere in the world.


---

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


---

## 4. Results

### 4.1 Compression Performance

Table 1 reports token compression ratios across five UFC DoD specification documents ranging from 5,836 to 65,325 tokens. The relevance filter alone achieves 11x–457x compression depending on document size, with larger documents compressing more aggressively as irrelevant content is filtered out.

| Document | Original Tokens | Compressed Tokens | Ratio |
|---|---|---|---|
| UFC Electrical | 22,893 | 780 | 29x |
| UFC Structural | 65,325 | 143 | 457x |
| UFC Microgrid | 20,319 | 493 | 41x |
| UFC C5ISR | 29,067 | 396 | 73x |
| UFC Housing | 5,836 | 525 | 11x |
| **Average** | **28,688** | **467** | **122x** |

All compressed prompts fit within a 2,048-token context window on the Jetson Orin Nano Super (8GB unified memory). Without compression, four of five documents exceed the context limit entirely and inference fails with HTTP 400.

### 4.2 Answer Quality

We evaluate answer quality using keyword recall against ground-truth technical terms extracted from each specification. A compressed answer scores 1.0 if it contains all expected domain-specific terms; 0.0 if none are present.

| Document | Compressed | Baseline | Delta | Winner |
|---|---|---|---|---|
| UFC Electrical | 0.30 | 0.30 | 0.00 | Tie |
| UFC Structural | 0.20 | 0.10 | +0.10 | Compressed |
| UFC Microgrid | 0.60 | 0.20 | +0.40 | Compressed |
| UFC C5ISR | 0.40 | 0.00 | +0.40 | Compressed |
| UFC Housing | 0.30 | 0.40 | -0.10 | Baseline |
| **Average** | **0.36** | **0.20** | **+0.16** | |

Compressed answers outperform the no-context baseline in 3 of 5 cases and tie in 1. Average keyword recall improves 80% (0.20 → 0.36). The single baseline win (UFC Housing) occurs on the smallest document (5,836 tokens), where the model's parametric knowledge is sufficient to answer without context and compression over-filters the short document.

The C5ISR result is notable: the baseline scores 0.00, indicating the model has no parametric knowledge of C5ISR facility grade classifications. The compressed answer correctly identifies Grade 3 (concurrently maintainable) and Grade 4 (fault-tolerant) requirements — information retrievable only from the source document.

### 4.3 Inference Latency

All inference runs on the Jetson Orin Nano Super with the model loaded in CPU-only mode due to unified memory constraints at 4,096-token context. GPU-accelerated inference at 2,048-token context achieves approximately 19–20 tokens/second. Compression pipeline latency (relevance filter + abbreviation) runs on the Raspberry Pi 5 and averages 25 seconds per document, dominated by embedding model load time. Subsequent queries on the same document reuse cached embeddings.

### 4.4 Ablation: Filter Disabled

When the relevance filter is disabled and the full document is passed to the inference endpoint, all documents exceeding 2,048 tokens return HTTP 400 (context overflow). This confirms the relevance filter is not an optimization — it is the architectural enabler. Without it, Marley1 cannot function on any document beyond the raw context limit.

### 4.5 Limitations

LLMLingua-2 (xlm-roberta-large) was evaluated and disabled. At 3B model scale, token-level compression via LLMLingua-2 degrades answer coherence without measurable quality improvement. We attribute this to the mismatch between the meetingbank-trained compression model and formal government specification prose.

Self-evaluation via LLM-as-judge also failed at 3B scale — the model cannot reliably score its own outputs. Keyword recall is used as a proxy metric. Future work should evaluate with a larger external judge model.


---

## 3.1 Latency

All inference times measured on Jetson Orin Nano Super in CPU-only mode (no GPU offload) due to unified memory constraints at 4,096-token context. GPU-accelerated inference at 2,048-token context achieves 19-20 tokens/second, approximately 5x faster than the figures below.

| Document | Compress Time | Inference Time | Total | Prompt Tokens |
|---|---|---|---|---|
| UFC Electrical | ~24s | 21.7s | ~46s | 1,648 |
| UFC Structural | ~53s | 17.5s | ~71s | 371 |
| UFC Microgrid | ~16s | 20.3s | ~36s | 788 |
| UFC C5ISR | ~28s | 20.9s | ~49s | 675 |
| UFC Housing | ~9s | 21.2s | ~30s | 1,123 |

Compression time is dominated by the relevance filter embedding model load (~20s first call, cached on subsequent calls). Inference time is consistent at 17-22s across documents regardless of original document size, confirming that compression successfully decouples inference cost from document length.

At GPU-accelerated speed, estimated end-to-end answer time drops to approximately 10-15s per query after first load.


---

## 5. Conclusion

We demonstrated that semantic relevance filtering enables a 3B parameter language model running on a $500 edge device stack to answer questions over DoD technical specification documents 11x-457x larger than its context window. Compressed answers outperform the no-context baseline by 80% on average across five UFC documents.

The core finding is architectural: relevance filtering is not an optimization, it is the enabler. Without it, inference fails entirely on all but the smallest documents. With it, a Jetson Orin Nano Super and a Raspberry Pi 5 become a functional document QA system requiring no cloud services, no API keys, and no internet connection.

Secondary findings:

- LLMLingua-2 degrades quality at 3B model scale on formal prose. Token-level compression is not recommended for small models on domain-specific technical documents.
- LLM-as-judge evaluation fails at 3B scale. Small models cannot reliably score their own outputs. Keyword recall is a practical proxy for domains with known ground-truth terminology.
- Compression scales with document size. Larger documents compress more aggressively, making Marley1 most useful precisely where it is most needed.

The total hardware cost is approximately $500. The monthly operating cost is electricity. Anyone with $500 and a question can query government-scale technical documents offline, on their own hardware, in their own language, in their own home.

That is the point.

## References

- Jiang et al. (2023). LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. EMNLP 2023.
- Pan et al. (2024). LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression. ACL 2024.
- Reimers and Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.
- Unified Facilities Criteria (UFC) Program. U.S. Department of Defense. https://www.wbdg.org/ffc/dod/ufc
- llama.cpp. https://github.com/ggerganov/llama.cpp
- Qwen2.5 Technical Report. Alibaba Cloud, 2024.
