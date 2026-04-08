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
