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
