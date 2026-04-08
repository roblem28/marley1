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
