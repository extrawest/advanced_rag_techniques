"""
Hybrid Search Tutorial - BM25 + Ensemble Retrieval

This script demonstrates how to create a hybrid search system using LangChain by:
1. Creating a BM25 sparse retriever
2. Creating a FAISS dense vector retriever
3. Combining them with an EnsembleRetriever for improved results
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings


def create_sample_texts() -> List[str]:
    """
    Create a sample list of text documents for demonstration.
    
    Returns:
        List[str]: Sample text documents
    """
    return [
        "I like apples",
        "I like oranges",
        "Apples and oranges are fruits",
        "I like computers by Apple",
        "I love fruit juice"
    ]


def create_bm25_retriever(texts: List[str], k: int = 2) -> BM25Retriever:
    """
    Create a BM25 sparse retriever from a list of texts.
    
    BM25 is a sparse retrieval algorithm that uses term frequency-inverse document frequency.
    It works well with exact keyword matching.
    
    Args:
        texts: List of text documents
        k: Number of documents to retrieve
        
    Returns:
        BM25Retriever: Configured BM25 retriever
    """
    retriever = BM25Retriever.from_texts(texts)
    retriever.k = k
    return retriever


def create_vector_retriever(texts: List[str], k: int = 2) -> BaseRetriever:
    """
    Create a dense vector retriever using FAISS.
    
    FAISS is an efficient similarity search library that uses embeddings
    to find semantically similar documents.
    
    Args:
        texts: List of text documents
        k: Number of documents to retrieve
        
    Returns:
        BaseRetriever: Configured FAISS retriever
    """
    embedding = OpenAIEmbeddings()

    faiss_vectorstore = FAISS.from_texts(texts, embedding)
    return faiss_vectorstore.as_retriever(search_kwargs={"k": k})


def create_ensemble_retriever(
    retrievers: List[BaseRetriever],
    weights: Optional[List[float]] = None
) -> EnsembleRetriever:
    """
    Create an ensemble retriever that combines multiple retrievers.
    
    The ensemble retriever combines results from multiple retrievers,
    allowing for both keyword and semantic matching.
    
    Args:
        retrievers: List of retrievers to combine
        weights: Optional weights for each retriever (must sum to 1)
        
    Returns:
        EnsembleRetriever: Configured ensemble retriever
    """
    if weights is None:
        weights = [1/len(retrievers)] * len(retrievers)
        
    return EnsembleRetriever(retrievers=retrievers, weights=weights)


def query_and_print_results(retriever: BaseRetriever, query: str, retriever_name: str) -> List[Document]:
    """
    Query a retriever and print the results in a formatted way.
    
    Args:
        retriever: The retriever to use
        query: Query string
        retriever_name: Name of the retriever for display
        
    Returns:
        List[Document]: Retrieved documents
    """
    print(f"\n=== {retriever_name} Results for: '{query}' ===")
    
    results = retriever.invoke(query)
    
    if not results:
        print("No results found.")
        return []
    
    for i, doc in enumerate(results):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content}")
        if hasattr(doc, 'metadata') and doc.metadata:
            print(f"Metadata: {doc.metadata}")
            
    return results


def compare_retrievers(
    query: str,
    bm25_retriever: BM25Retriever,
    vector_retriever: BaseRetriever,
    ensemble_retriever: EnsembleRetriever
) -> None:
    """
    Compare results from different retrievers for the same query.
    
    Args:
        query: Query string to test
        bm25_retriever: BM25 sparse retriever
        vector_retriever: Vector-based dense retriever
        ensemble_retriever: Combined ensemble retriever
    """
    print(f"\n\n==== Comparing Retrievers for Query: '{query}' ====")

    bm25_results = query_and_print_results(bm25_retriever, query, "BM25 Retriever (Sparse)")
    vector_results = query_and_print_results(vector_retriever, query, "FAISS Retriever (Dense)")
    ensemble_results = query_and_print_results(ensemble_retriever, query, "Ensemble Retriever (Hybrid)")

    print("\n=== Retrieval Summary ===")
    print(f"BM25 found {len(bm25_results)} documents")
    print(f"FAISS found {len(vector_results)} documents")
    print(f"Ensemble found {len(ensemble_results)} documents")


def main() -> None:
    """Execute the hybrid search tutorial."""
    print("Hybrid Search Tutorial - BM25 + Ensemble")
    print("---------------------------------------")

    texts = create_sample_texts()
    print(f"Created {len(texts)} sample documents:")
    for i, text in enumerate(texts):
        print(f"  {i+1}. {text}")

    retries_number = 2
    bm25_retriever = create_bm25_retriever(texts, retries_number)
    vector_retriever = create_vector_retriever(texts, retries_number)

    ensemble_retriever = create_ensemble_retriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    compare_retrievers("Apple", bm25_retriever, vector_retriever, ensemble_retriever)
    compare_retrievers("a green fruit", bm25_retriever, vector_retriever, ensemble_retriever)
    compare_retrievers("Apple Phones", bm25_retriever, vector_retriever, ensemble_retriever)


if __name__ == "__main__":
    main()
