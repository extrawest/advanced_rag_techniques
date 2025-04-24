"""
Self-Query Retriever Tutorial

This script demonstrates how to use the SelfQueryRetriever from LangChain
to retrieve documents based on natural language queries and metadata filters.

The SelfQueryRetriever can:
1. Understand natural language queries
2. Apply metadata filters automatically
3. Combine semantic search with structured metadata filtering
4. Limit the number of returned documents
"""

from typing import List
import sqlite3

sqlite3.sqlite_version_info = (3, 45, 0)
sqlite3.sqlite_version = "3.45.0"

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


def create_sample_wine_data() -> List[Document]:
    """
    Create a sample dataset of wine documents with metadata.
    
    Returns:
        List[Document]: List of wine documents with metadata
    """
    return [
        Document(
            page_content="Complex, layered, rich red with dark fruit flavors",
            metadata={
                "name": "Opus One",
                "year": 2018,
                "rating": 96,
                "grape": "Cabernet Sauvignon",
                "color": "red",
                "country": "USA"
            },
        ),
        Document(
            page_content="Luxurious, sweet wine with flavors of honey, apricot, and peach",
            metadata={
                "name": "Château d'Yquem",
                "year": 2015,
                "rating": 98,
                "grape": "Sémillon",
                "color": "white",
                "country": "France"
            },
        ),
        Document(
            page_content="Full-bodied red with notes of black fruit and spice",
            metadata={
                "name": "Penfolds Grange",
                "year": 2017,
                "rating": 97,
                "grape": "Shiraz",
                "color": "red",
                "country": "Australia"
            },
        ),
        Document(
            page_content="Elegant, balanced red with herbal and berry nuances",
            metadata={
                "name": "Sassicaia",
                "year": 2016,
                "rating": 95,
                "grape": "Cabernet Franc",
                "color": "red",
                "country": "Italy"
            },
        ),
        Document(
            page_content="Highly sought-after Pinot Noir with red fruit and earthy notes",
            metadata={
                "name": "Domaine de la Romanée-Conti",
                "year": 2018,
                "rating": 100,
                "grape": "Pinot Noir",
                "color": "red",
                "country": "France"
            },
        ),
        Document(
            page_content="Crisp white with tropical fruit and citrus flavors",
            metadata={
                "name": "Cloudy Bay",
                "year": 2021,
                "rating": 92,
                "grape": "Sauvignon Blanc",
                "color": "white",
                "country": "New Zealand"
            },
        ),
        Document(
            page_content="Rich, complex Champagne with notes of brioche and citrus",
            metadata={
                "name": "Krug Grande Cuvée",
                "year": 2010,
                "rating": 93,
                "grape": "Chardonnay blend",
                "color": "sparkling",
                "country": "New Zealand"
            },
        ),
        Document(
            page_content="Intense, dark fruit flavors with hints of chocolate",
            metadata={
                "name": "Caymus Special Selection",
                "year": 2018,
                "rating": 96,
                "grape": "Cabernet Sauvignon",
                "color": "red",
                "country": "USA"
            },
        ),
        Document(
            page_content="Exotic, aromatic white with stone fruit and floral notes",
            metadata={
                "name": "Jermann Vintage Tunina",
                "year": 2020,
                "rating": 91,
                "grape": "Sauvignon Blanc blend",
                "color": "white",
                "country": "Italy"
            },
        ),
    ]


def create_metadata_field_info() -> List[AttributeInfo]:
    """
    Define metadata field information for the self-query retriever.
    
    Returns:
        List[AttributeInfo]: Metadata field definitions
    """
    return [
        AttributeInfo(
            name="grape",
            description="The grape used to make the wine",
            type="string or list[string]",
        ),
        AttributeInfo(
            name="name",
            description="The name of the wine",
            type="string or list[string]",
        ),
        AttributeInfo(
            name="color",
            description="The color of the wine",
            type="string or list[string]",
        ),
        AttributeInfo(
            name="year",
            description="The year the wine was released",
            type="integer",
        ),
        AttributeInfo(
            name="country",
            description="The name of the country the wine comes from",
            type="string",
        ),
        AttributeInfo(
            name="rating",
            description="The Robert Parker rating for the wine 0-100",
            type="integer",
        ),
    ]


def setup_vectorstore(docs: List[Document]) -> Chroma:
    """
    Create and set up a vector store with the provided documents.
    
    Args:
        docs: List of documents to index
    
    Returns:
        Chroma: Configured vector store
    """
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(docs, embeddings)


def create_self_query_retriever(
    vectorstore: Chroma,
    metadata_field_info: List[AttributeInfo],
    enable_limit: bool = False
) -> SelfQueryRetriever:
    """
    Create a self-query retriever based on the provided vector store and metadata fields.
    
    Args:
        vectorstore: Vector store containing the documents
        metadata_field_info: Information about metadata fields
        enable_limit: Whether to enable the limit parameter to control number of results
    
    Returns:
        SelfQueryRetriever: Configured retriever
    """
    llm = OpenAI(temperature=0)
    document_content_description = "Brief description of the wine"

    return SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        enable_limit=enable_limit,
        verbose=True
    )


def query_and_print_results(retriever: SelfQueryRetriever, query: str) -> None:
    """
    Execute a query and print the results in a formatted way.
    
    Args:
        retriever: The retriever to use
        query: Query string to execute
    """
    print(f"\n=== Results for: {query} ===")
    
    results = retriever.invoke(query)
    
    if not results:
        print("No results found.")
        return
    
    for i, doc in enumerate(results):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")


def demo_basic_queries(retriever: SelfQueryRetriever) -> None:
    """
    Demonstrate basic queries with the self-query retriever.
    
    Args:
        retriever: Configured retriever
    """
    print("\n--- Basic Query Examples ---")

    query_and_print_results(retriever, "What are some red wines")
    query_and_print_results(retriever, "I want a wine that has fruity nodes")

    query_and_print_results(
        retriever,
        "I want a wine that has fruity nodes and has a rating above 97"
    )

    query_and_print_results(retriever, "What wines come from Italy?")

    query_and_print_results(
        retriever,
        "What's a wine after 2015 but before 2020 that's all earthy"
    )


def demo_limit_queries(retriever: SelfQueryRetriever) -> None:
    """
    Demonstrate queries with limit parameter.
    
    Args:
        retriever: Configured retriever with limit enabled
    """
    print("\n--- Limit Query Examples ---")

    query_and_print_results(
        retriever,
        "what are two that have a rating above 97"
    )

    query_and_print_results(
        retriever,
        "what are two wines that come from australia or New zealand"
    )


def main() -> None:
    """Execute the SelfQueryRetriever tutorial."""
    print("Self-Query Retriever Tutorial")
    print("----------------------------")

    docs = create_sample_wine_data()

    vectorstore = setup_vectorstore(docs)

    metadata_field_info = create_metadata_field_info()

    basic_retriever = create_self_query_retriever(
        vectorstore,
        metadata_field_info,
        enable_limit=False
    )
    demo_basic_queries(basic_retriever)

    limit_retriever = create_self_query_retriever(
        vectorstore,
        metadata_field_info,
        enable_limit=True
    )
    demo_limit_queries(limit_retriever)


if __name__ == "__main__":
    main()
