# 🚀 LangChain RAG Tips and Tricks

This repository contains a series of lessons demonstrating advanced Retrieval-Augmented Generation (RAG) techniques using LangChain. 
Each lesson focuses on a specific approach to improve retrieval quality and efficiency.

## 📚 Lessons Overview

### Lesson 1: Self-Query Retriever

This lesson demonstrates how to use the SelfQueryRetriever to automatically convert natural language queries into semantic searches and metadata filters.

**Key Components:**
- 🧠 SelfQueryRetriever for intelligent query parsing
- 🏷️ Metadata field definitions for structured filtering
- 🔍 Vector search with metadata constraints
- 🔢 Result limiting and pagination

**What You'll Learn:**
- Creating and structuring a vector store with metadata
- Setting up metadata field definitions for intelligent filtering
- Configuring and using SelfQueryRetriever for semantic search
- Combining vector search with metadata filtering
- Limiting query results using the limit parameter

### Lesson 2: Parent Document Retriever

This lesson explores the ParentDocumentRetriever which allows searching on small chunks but returning whole documents or larger chunks.

**Key Components:**
- 📄 ParentDocumentRetriever for hierarchical document access
- 📦 Dual storage for small and large document fragments
- 🔄 Small-to-large chunk mapping system
- 🤖 Integration with LLM for complete QA workflow

**What You'll Learn:**
- Creating a system that allows searching documents by small fragments
- Setting up storage for small and large document fragments
- Two modes of ParentDocumentRetriever operation:
  - Returning full documents based on small chunk matches
  - Returning larger chunks based on small chunk matches
- Integrating the retriever with an LLM for a complete QA system

### Lesson 3: Hybrid Search (BM25 + Ensemble Retrieval)

This lesson demonstrates creating a hybrid search system that combines sparse (BM25) and dense (FAISS) vector representations using an EnsembleRetriever.

**Key Components:**
- 📚 BM25 retriever for keyword-based search
- 🧮 FAISS vector retriever for semantic search
- 🔄 EnsembleRetriever for combining search strategies
- ⚖️ Result weighting and score normalization
- 📊 Comparative performance analysis

**What You'll Learn:**
- Creating a BM25 retriever for efficient keyword search
- Creating a dense vector FAISS retriever for semantic search
- Combining both retrievers with EnsembleRetriever
- Weighting results for optimal hybrid search performance
- Comparing the effectiveness of different search approaches

### Lesson 4: Contextual Compression and Filtering

This lesson focuses on improving the quality of context for LLMs through contextual compression and filtering to retrieve only relevant information.

**Key Components:**
- 📑 SentenceWindowNodeParser for context-aware text chunking
- 🔄 MetadataReplacementPostProcessor for enhancing retrieval context
- 🏆 SentenceTransformerRerank for improving result relevance
- 📊 Comparative analysis of different window sizes

**What You'll Learn:**
- How sentence window parsing creates contextual overlaps between chunks
- How to configure optimal window sizes for different content types
- How reranking can enhance the quality of retrieved passages
- How metadata replacement provides more complete context
- How to compare different window size configurations (1 vs 3)
- The impact of context window size on RAG performance metrics

### Lesson 5: Hypothetical Document Embeddings (HyDE)

This lesson explores the Hypothetical Document Embeddings (HyDE) technique which improves search by creating hypothetical answers to questions.

**Key Components:**
- 💭 LLM-generated hypothetical documents
- 🔄 Query-to-document transformation
- 🧩 Multiple generation variants for robustness
- 📊 Comparative evaluation against standard search
- 🧠 BGE embedding model integration

**What You'll Learn:**
- Creating hypothetical documents using an LLM
- Embedding these hypothetical documents instead of direct queries
- Working with different HyDE implementation variants:
  - Basic HyDE with predefined templates
  - HyDE with multiple generations for improved results
  - HyDE with custom prompts for specific tasks
- Comparing HyDE search with standard vector search
- Working with BGE embeddings for improved result quality

### Lesson 6: RAGFusion

This lesson demonstrates the RAGFusion technique which generates multiple search queries from a single user query and combines the results.

**Key Components:**
- 🔄 Query generation for diverse search angles
- 📚 Multi-query document retrieval system
- 📊 Reciprocal Rank Fusion for result combination
- ⚡ Asynchronous execution pipeline
- 🧠 OpenAI and BGE embedding model support

**What You'll Learn:**
- Generating multiple search queries from a single user query
- Retrieving documents for each generated query
- Combining results using the Reciprocal Rank Fusion algorithm
- Setting up and using OpenAI and BGE embeddings
- Comparing standard RAG and RAGFusion
- Asynchronous query execution for improved performance
- Serializing and deserializing documents for efficient ranking

## 🛠️ Requirements

The code in these lessons uses various components from:
- langchain_core
- langchain_community
- langchain_openai
- langchain_text_splitters
- FAISS
- Chroma
- HuggingFace models for embeddings

## 🚀 Usage

Each lesson is a standalone Python script that can be run independently. 
Make sure to install the required dependencies and configure any necessary API keys for external services like OpenAI.