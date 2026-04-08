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
