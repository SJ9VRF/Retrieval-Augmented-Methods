# Retrieval Augmented Generation Methods

![Screenshot_2025-01-25_at_12 23 31_PM-removebg-preview](https://github.com/user-attachments/assets/608762ca-ee96-4f22-a75d-13074f11e6f4)


## Traditional RAG
### Paper Title
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
### Link
[https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
### Summary
Combines pre-trained parametric and non-parametric memory for language generation.
### Approach Details
- Uses a pre-trained seq2seq model as parametric memory.
- Employs a dense vector index of Wikipedia as non-parametric memory.
- Utilizes a pre-trained neural retriever for accessing the knowledge base.
### Performance
Sets state-of-the-art on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures.
### Pros
- Improves factual accuracy and knowledge integration.
- Allows for continuous knowledge updates.
- Enhances specificity and diversity in language generation.
### Cons
- May introduce latency due to retrieval process.
- Performance depends on the quality of retrieved documents.

## LongRAG
### Paper Title
LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs
### Summary
Introduces a "long retriever" and "long reader" framework to process longer context units.
### Approach Details
- Processes Wikipedia into 4K-token units (30 times longer than traditional methods).
- Utilizes a long-context LLM for zero-shot answer extraction.
### Performance
- Answer recall@1 = 71% on NQ.
- Answer recall@2 = 72% on HotpotQA (full-wiki).
- EM scores: 62.7% on NQ and 64.3% on HotpotQA.
### Pros
- Reduces the number of retrieval units significantly.
- Achieves higher retrieval scores.
- Improves performance on complex queries.
### Cons
- Requires long-context LLMs, which may be more resource-intensive.
- May struggle with very specific, localized information within long passages.

## GAR-meets-RAG
### Paper Title
GAR-meets-RAG Paradigm for Zero-Shot Information Retrieval
### Summary
Combines generation-augmented retrieval (GAR) and retrieval-augmented generation (RAG) for zero-shot information retrieval.
### Approach Details
- Iteratively enhances both retrieval and rewriting stages.
- Integrates generative capabilities of LLMs with embedding-based retrieval.
### Performance
Achieves up to 17% relative gains over prior state-of-the-art results on BEIR and TREC-DL datasets.
### Pros
- Improves both recall and precision in document ranking.
- Effective in zero-shot scenarios without domain-specific training data.
- Enhances performance on passage retrieval tasks.
### Cons
- May require complex implementation and fine-tuning.
- Potential increased computational cost due to iterative process.

## LLM-Embedder
### Paper Title
Retrieve Anything To Augment Large Language Models
### Summary
A unified model designed to optimize retrieval for diverse LLM needs with a single model.
### Approach Details
- Utilizes optimized training methodologies:
  - Reward formulation.
  - Stabilized knowledge distillation.
  - Multi-task fine-tuning.
  - Homogeneous negative sampling.
### Performance
Demonstrates outstanding empirical performance (specific metrics not provided in search results).
### Pros
- Addresses varied semantic relationships in different retrieval tasks.
- Offers a unified solution for enhancing LLM capabilities through retrieval augmentation.
### Cons
- May require complex training process.
- Potential challenges in balancing performance across diverse retrieval tasks.

## RAGElo
### Paper Title
Evaluating RAG-Fusion with RAGElo: an Automated Elo-based Framework
### Summary
An automated Elo-based evaluation framework for RAG systems.
### Approach Details
- Uses LLMs to generate synthetic queries.
- Assesses retrieved documents and answers with LLM-based judging.
- Ranks RAG variants using an automated Elo-based competition.
### Performance
Not specified in search results.
### Pros
- Addresses challenges in evaluating domain-specific RAG QA systems.
- Provides a comprehensive evaluation framework without human annotations.
### Cons
- Relies on LLMs for evaluation, which may introduce biases.
- May not fully capture real-world performance nuances.

## Fusion-in-Decoder (FiD)
### Paper Title
Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering
### Link
[https://arxiv.org/abs/2007.01282](https://arxiv.org/abs/2007.01282)
### Summary
FiD improves upon RAG by allowing the language model to attend over all retrieved documents simultaneously during the decoding phase.
### Approach Details
- Employs a powerful decoder that processes information from multiple documents together.
- Enhances the model's ability to synthesize information from multiple sources.
- Generates more coherent and contextually rich outputs.
### Performance
Achieves state-of-the-art results on multiple open-domain QA datasets (specific metrics not provided in the query).
### Pros
- Better at handling multiple pieces of evidence.
- Produces more contextually aware responses.
- Improves information synthesis from diverse sources.
### Cons
- More resource-intensive than traditional RAG, particularly in terms of memory usage during decoding.
- May require more powerful hardware for optimal performance.

## kNN-LM (k-Nearest Neighbors Language Model)
### Paper Title
Generalization through Memorization: Nearest Neighbor Language Models
### Link
[https://arxiv.org/abs/1911.00172](https://arxiv.org/abs/1911.00172)
### Summary
Augments a pre-trained language model with a k-nearest neighbor (kNN) search over a datastore of precomputed contextual representations.
### Approach Details
- Builds a datastore of precomputed contextual representations from a high-quality dataset.
- Performs kNN search at runtime to find the most similar past examples.
- Dynamically adapts model predictions based on retrieved examples.
### Performance
Improves perplexity and other language modeling metrics across various domains (specific numbers not provided in the query).
### Pros
- Enhances model adaptability to new information and contexts without full retraining.
- Allows for easy integration of domain-specific knowledge.
- Improves performance on domain-specific tasks.
### Cons
- Sensitive to the quality of the datastore.
- Can result in high latency due to the need for runtime retrieval.
- Requires careful management of the datastore for optimal performance.

## RETRO (Retrieval-Enhanced Transformer)
### Paper Title
Improving language models by retrieving from trillions of tokens
### Link
[https://arxiv.org/abs/2112.04426](https://arxiv.org/abs/2112.04426)
### Summary
RETRO uses a large-scale retrieval mechanism where chunks of text are pre-encoded and stored, and these chunks are retrieved and concatenated to the input as additional context during generation.
### Approach Details
- Pre-encodes and stores large chunks of text in a massive datastore.
- Retrieves relevant chunks during generation and concatenates them to the input.
- Employs a specialized architecture to process the retrieved chunks alongside the input.
### Performance
Achieves competitive performance with much larger models while using significantly less computing power for training (specific metrics not provided in the query).
### Pros
- Leverages vast amounts of data for better knowledge coverage and contextual understanding.
- Reduces the need for extremely large model parameters.
- Enables efficient scaling of language models.
### Cons
- Managing and updating the datastore can be complex and computationally expensive.
- May introduce latency due to the retrieval process.
- Requires careful design of the retrieval mechanism to ensure relevance.
