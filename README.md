# LangChain Advanced RAG Tips and Tricks

This repository contains a series of lessons demonstrating advanced Retrieval-Augmented Generation (RAG) techniques using LangChain. 
Each lesson focuses on a specific approach to improve retrieval quality and efficiency.

## Lessons Overview

### Lesson 1: Self-Query Retriever

This lesson demonstrates how to use the SelfQueryRetriever to automatically convert natural language queries into semantic searches and metadata filters.

**What you'll learn:**
- Creating and structuring a vector store with metadata
- Setting up metadata field definitions for intelligent filtering
- Configuring and using SelfQueryRetriever for semantic search
- Combining vector search with metadata filtering
- Limiting query results using the limit parameter

### Lesson 2: Parent Document Retriever

This lesson explores the ParentDocumentRetriever which allows searching on small chunks but returning whole documents or larger chunks.

**What you'll learn:**
- Creating a system that allows searching documents by small fragments
- Setting up storage for small and large document fragments
- Two modes of ParentDocumentRetriever operation:
  - Returning full documents based on small chunk matches
  - Returning larger chunks based on small chunk matches
- Integrating the retriever with an LLM for a complete QA system

### Lesson 3: Hybrid Search (BM25 + Ensemble Retrieval)

This lesson demonstrates creating a hybrid search system that combines sparse (BM25) and dense (FAISS) vector representations using an EnsembleRetriever.

**What you'll learn:**
- Creating a BM25 retriever for efficient keyword search
- Creating a dense vector FAISS retriever for semantic search
- Combining both retrievers with EnsembleRetriever
- Weighting results for optimal hybrid search performance
- Comparing the effectiveness of different search approaches

### Lesson 4: Contextual Compression and Filtering

This lesson focuses on improving the quality of context for LLMs through contextual compression and filtering to retrieve only relevant information.

**What you'll learn:**
- Extracting relevant content from documents using an LLM
- Filtering documents using LLM decisions (yes/no)
- Filtering documents based on embedding similarity threshold
- Creating document transformation pipelines for complex processing
- Combining various compression and filtering methods
- Using asynchronous operations for efficient I/O processing

### Lesson 5: Hypothetical Document Embeddings (HyDE)

This lesson explores the Hypothetical Document Embeddings (HyDE) technique which improves search by creating hypothetical answers to questions.

**What you'll learn:**
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

**What you'll learn:**
- Generating multiple search queries from a single user query
- Retrieving documents for each generated query
- Combining results using the Reciprocal Rank Fusion algorithm
- Setting up and using OpenAI and BGE embeddings
- Comparing standard RAG and RAGFusion
- Asynchronous query execution for improved performance
- Serializing and deserializing documents for efficient ranking

## Requirements

The code in these lessons uses various components from:
- langchain_core
- langchain_community
- langchain_openai
- langchain_text_splitters
- FAISS
- Chroma
- HuggingFace models for embeddings

## Usage

Each lesson is a standalone Python script that can be run independently. 
Make sure to install the required dependencies and configure any necessary API keys for external services like OpenAI.