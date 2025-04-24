"""
Parent Document Retriever Tutorial

This script demonstrates two ways to use the ParentDocumentRetriever:
1. Return full docs from smaller chunks lookup
2. Return bigger chunks from smaller chunks lookup
"""

from typing import List, Any
import os

from pathlib import Path
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


def load_documents() -> List[Document]:
    """
    Load documents from the blog_posts directory.
    
    Returns:
        List[Document]: List of loaded documents
    """
    blog_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "blog_posts"

    file_paths = [
        blog_dir / "blog.langchain.dev_announcing-langsmith_.txt",
        blog_dir / "blog.langchain.dev_benchmarking-question-answering-over-csv-data_.txt"
    ]

    valid_paths = [path for path in file_paths if path.exists()]
    
    if not valid_paths:
        print("Warning: None of the specified blog post files were found.")
        print(f"Expected files in: {blog_dir}")
        print("Available files:")
        if blog_dir.exists():
            for file in blog_dir.glob("*.txt"):
                print(f"  - {file.name}")
        else:
            print(f"Blog posts directory not found: {blog_dir}")
        return []

    docs = []
    for file_path in valid_paths:
        print(f"Loading: {file_path}")
        loader = TextLoader(str(file_path))
        docs.extend(loader.load())
    
    print(f"Loaded {len(docs)} documents")
    return docs


def setup_embeddings() -> Any:
    """
    Setup and return embedding model.
    
    Returns:
        Any: Embedding model instance
    """
    try:
        model_name = "BAAI/bge-small-en-v1.5"
        encode_kwargs = {'normalize_embeddings': True}

        device = 'cpu'
        
        return HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs=encode_kwargs
        )
    except Exception as e:
        print(f"Failed to initialize HuggingFace embeddings: {e}")
        print("Falling back to OpenAI embeddings")
        return OpenAIEmbeddings()


def create_full_doc_retriever(docs: List[Document], embeddings: Any) -> ParentDocumentRetriever:
    """
    Create a retriever that returns full documents based on small chunk matching.
    
    Args:
        docs: List of documents to index
        embeddings: Embedding model to use
        
    Returns:
        ParentDocumentRetriever: Configured retriever
    """
    print("\n=== Setting up Full Document Retriever ===")

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    vectorstore = Chroma(
        collection_name="full_documents",
        embedding_function=embeddings
    )

    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )

    retriever.add_documents(docs)
    
    print(f"Added {len(docs)} documents to the retriever")
    print(f"Created {len(list(store.yield_keys()))} parent document keys in storage")
    
    return retriever


def create_big_chunks_retriever(docs: List[Document], embeddings: Any) -> ParentDocumentRetriever:
    """
    Create a retriever that returns big chunks instead of full documents.
    
    Args:
        docs: List of documents to index
        embeddings: Embedding model to use
        
    Returns:
        ParentDocumentRetriever: Configured retriever
    """
    print("\n=== Setting up Big Chunks Retriever ===")

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    vectorstore = Chroma(
        collection_name="split_parents", 
        embedding_function=embeddings
    )

    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    retriever.add_documents(docs)
    
    print(f"Added {len(docs)} documents to the retriever")
    print(f"Created {len(list(store.yield_keys()))} parent document keys in storage")
    
    return retriever


def test_retriever(retriever: ParentDocumentRetriever, query: str, retriever_name: str) -> List[Document]:
    """
    Test a retriever with a query and display the results.
    
    Args:
        retriever: The retriever to test
        query: Query string to use
        retriever_name: Name of the retriever for display
        
    Returns:
        List[Document]: The retrieved documents
    """
    print(f"\n=== Testing {retriever_name} with query: '{query}' ===")

    sub_docs = retriever.vectorstore.similarity_search(query, k=2)
    
    print(f"\n* Vectorstore returned {len(sub_docs)} small chunks:")
    for i, doc in enumerate(sub_docs):
        print(f"\nSmall Chunk {i+1} ({len(doc.page_content)} chars):")
        print(f"Content snippet: {doc.page_content[:150]}...")

    retrieved_docs = retriever.get_relevant_documents(query)
    
    print(f"\n* ParentDocumentRetriever returned {len(retrieved_docs)} documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"\nParent Document {i+1} ({len(doc.page_content)} chars):")
        print(f"Content snippet: {doc.page_content[:150]}...")
    
    return retrieved_docs


def run_qa(retriever: ParentDocumentRetriever, query: str) -> None:
    """
    Run a QA chain with the retriever and display results.
    
    Args:
        retriever: Retriever to use for QA
        query: Question to answer
    """
    try:
        print(f"\n=== Running QA for query: '{query}' ===")

        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0),
            chain_type="stuff",
            retriever=retriever
        )

        result = qa.invoke(query)
        
        print("\n* QA Result:")
        print(result['result'])
    except Exception as e:
        print(f"Error running QA: {e}")
        print("Note: You may need to set your OPENAI_API_KEY environment variable")


def main() -> None:
    """Run the parent document retriever tutorial."""
    print("Parent Document Retriever Tutorial")
    print("----------------------------------")

    docs = load_documents()
    if not docs:
        print("No documents loaded. Exiting.")
        return

    embeddings = setup_embeddings()

    full_doc_retriever = create_full_doc_retriever(docs, embeddings)

    big_chunks_retriever = create_big_chunks_retriever(docs, embeddings)

    query = "What is LangSmith?"

    full_docs = test_retriever(full_doc_retriever, query, "Full Document Retriever")

    big_chunks = test_retriever(big_chunks_retriever, query, "Big Chunks Retriever")

    if full_docs and big_chunks:
        print("\n=== Size Comparison ===")
        print(f"Full document size: {len(full_docs[0].page_content)} chars")
        print(f"Big chunk size: {len(big_chunks[0].page_content)} chars")
        ratio = len(full_docs[0].page_content) / len(big_chunks[0].page_content)
        print(f"Ratio: {ratio:.2f}x larger")

    run_qa(big_chunks_retriever, query)


if __name__ == "__main__":
    main()
