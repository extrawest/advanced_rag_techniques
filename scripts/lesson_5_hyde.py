"""
LangChain RAG Tips and Tricks 05 - Hypothetical Document Embeddings (HyDE)

This script demonstrates how to use Hypothetical Document Embeddings (HyDE) technique:
1. Create "hypothetical" answers to queries using an LLM
2. Embed these hypothetical documents instead of direct queries
3. Use the resulting embeddings for more effective document retrieval
"""

import os

from pathlib import Path

import warnings
from typing import List, Any

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from langchain_openai import OpenAI

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

from langchain_community.document_loaders import TextLoader

from langchain.chains import LLMChain
from langchain.chains import HypotheticalDocumentEmbedder


def initialize_bge_embeddings(
    model_name: str = "BAAI/bge-small-en-v1.5",
    device: str = "cpu"
) -> HuggingFaceBgeEmbeddings:
    """
    Initialize BGE embeddings model.
    
    Args:
        model_name: Name of the Hugging Face model to use
        device: Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        HuggingFaceBgeEmbeddings: Configured embeddings model
    """
    encode_kwargs = {'normalize_embeddings': True}
    
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs=encode_kwargs
    )


def create_basic_hyde_embeddings(
    llm: Any,
    base_embeddings: Any,
    prompt_key: str = "web_search"
) -> HypotheticalDocumentEmbedder:
    """
    Create basic HyDE embeddings with predefined prompts.
    
    Args:
        llm: Language model to generate hypothetical documents
        base_embeddings: Base embedding model for encoding documents
        prompt_key: Key for predefined prompt template
        
    Returns:
        HypotheticalDocumentEmbedder: Configured HyDE embeddings
    """
    return HypotheticalDocumentEmbedder.from_llm(
        llm,
        base_embeddings,
        prompt_key=prompt_key
    )


def create_multi_generation_hyde_embeddings(
    base_embeddings: Any,
    n: int = 4,
    best_of: int = 4,
    temperature: float = 0,
    prompt_key: str = "web_search"
) -> HypotheticalDocumentEmbedder:
    """
    Create HyDE embeddings that generate multiple documents and combine them.
    
    Args:
        base_embeddings: Base embedding model for encoding documents
        n: Number of completions to generate
        best_of: Number of completions to generate and select best from
        temperature: LLM temperature parameter
        prompt_key: Key for predefined prompt template
        
    Returns:
        HypotheticalDocumentEmbedder: Configured multi-generation HyDE
    """
    multi_llm = OpenAI(n=n, best_of=best_of, temperature=temperature)
    
    return HypotheticalDocumentEmbedder.from_llm(
        multi_llm,
        base_embeddings,
        prompt_key=prompt_key
    )


def create_custom_hyde_embeddings(
    llm: Any,
    base_embeddings: Any,
    template: str
) -> HypotheticalDocumentEmbedder:
    """
    Create HyDE embeddings with a custom prompt template.
    
    Args:
        llm: Language model to generate hypothetical documents
        base_embeddings: Base embedding model for encoding documents
        template: Custom prompt template string
        
    Returns:
        HypotheticalDocumentEmbedder: Configured custom HyDE
    """
    prompt = PromptTemplate(input_variables=["question"], template=template)

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    return HypotheticalDocumentEmbedder(
        llm_chain=llm_chain,
        base_embeddings=base_embeddings
    )


def load_and_split_documents(
    file_paths: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 0
) -> List[Document]:
    """
    Load documents from file paths and split them into chunks.
    
    Args:
        file_paths: List of file paths to load
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List[Document]: Split document chunks
    """
    docs = []
    for file_path in file_paths:
        try:
            loader = TextLoader(file_path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not docs:
        print("Warning: No documents were loaded.")
        return []

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return text_splitter.split_documents(docs)


def create_vector_store(
    documents: List[Document],
    embeddings: Any
) -> Chroma:
    """
    Create a vector store from documents and embeddings.
    
    Args:
        documents: List of documents to index
        embeddings: Embeddings model to use
        
    Returns:
        Chroma: Configured vector store
    """
    return Chroma.from_documents(documents, embeddings)


def perform_similarity_search(
    vector_store: Any,
    query: str,
    k: int = 4
) -> List[Document]:
    """
    Perform similarity search using a vector store.
    
    Args:
        vector_store: Vector store to search
        query: Query string
        k: Number of results to return
        
    Returns:
        List[Document]: Retrieved documents
    """
    return vector_store.similarity_search(query, k=k)


def print_search_results(
    query: str,
    docs: List[Document],
    search_type: str = "Standard search"
) -> None:
    """
    Print search results in a formatted way.
    
    Args:
        query: Original query
        docs: Retrieved documents
        search_type: Type of search performed
    """
    print(f"\n===== {search_type} Results for: '{query}' =====")
    
    if not docs:
        print("No results found.")
        return
    
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print(f"{doc.page_content[:300]}..." if len(doc.page_content) > 300 else doc.page_content)
        if hasattr(doc, 'metadata') and doc.metadata:
            print(f"Metadata: {doc.metadata}")


def demo_basic_hyde(
    base_embeddings: Any,
    temperature: float = 0
) -> None:
    """
    Demonstrate basic HyDE usage.
    
    Args:
        base_embeddings: Base embedding model
        temperature: LLM temperature parameter
    """
    print("\n\n== Demonstration: Basic HyDE with web_search prompt ==")

    llm = OpenAI(temperature=temperature)

    hyde_embeddings = create_basic_hyde_embeddings(
        llm=llm,
        base_embeddings=base_embeddings,
        prompt_key="web_search"
    )

    print("\nUsing prompt template:")
    print(hyde_embeddings.llm_chain.prompt.template)

    query = "What items does McDonalds make?"
    print(f"\nEmbedding query: '{query}'")

    result = hyde_embeddings.embed_query(query)

    print(f"Generated embedding vector of dimension: {len(result)}")


def demo_multi_generation_hyde(
    base_embeddings: Any
) -> None:
    """
    Demonstrate HyDE with multiple generations.
    
    Args:
        base_embeddings: Base embedding model
    """
    print("\n\n== Demonstration: HyDE with Multiple Generations ==")

    hyde_embeddings = create_multi_generation_hyde_embeddings(
        base_embeddings=base_embeddings,
        n=4,
        best_of=4
    )

    query = "What is McDonalds best selling item?"
    print(f"\nEmbedding query with multiple generations: '{query}'")

    result = hyde_embeddings.embed_query(query)

    print(f"Generated embedding vector of dimension: {len(result)}")


def demo_custom_prompt_hyde(
    base_embeddings: Any,
    temperature: float = 0
) -> None:
    """
    Demonstrate HyDE with custom prompt.
    
    Args:
        base_embeddings: Base embedding model
        temperature: LLM temperature parameter
    """
    print("\n\n== Demonstration: HyDE with Custom Prompt ==")

    template = """Please answer the user's question as a single food itemQuestion: {question} Answer:"""
    
    print(f"\nUsing custom prompt template:\n{template}")

    llm = OpenAI(temperature=temperature)

    hyde_embeddings = create_custom_hyde_embeddings(
        llm=llm,
        base_embeddings=base_embeddings,
        template=template
    )

    query = "What is McDonalds best selling item?"
    print(f"\nEmbedding query with custom prompt: '{query}'")

    result = hyde_embeddings.embed_query(query)

    print(f"Generated embedding vector of dimension: {len(result)}")


def demo_hyde_for_retrieval(
    base_embeddings: Any,
    temperature: float = 0
) -> None:
    """
    Demonstrate using HyDE for document retrieval.
    
    Args:
        base_embeddings: Base embedding model
        temperature: LLM temperature parameter
    """
    print("\n\n== Demonstration: Using HyDE for Document Retrieval ==")

    try:
        blog_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "blog_posts"

        document_paths = [
            str(blog_dir / "langchain_dev_test_info.txt")
        ]
        
        documents = load_and_split_documents(
            file_paths=document_paths,
            chunk_size=1000,
            chunk_overlap=0
        )
        
        if not documents:
            raise ValueError("No documents loaded")
            
    except Exception as e:
        print(f"Error loading documents: {e}")
        print("Using sample documents instead.")

        documents = create_sample_documents()
    
    print(f"Working with {len(documents)} document chunks.")

    template = """Please answer the user's question as related to Large Language Models Question: {question} Answer:"""

    llm = OpenAI(temperature=temperature)

    hyde_embeddings = create_custom_hyde_embeddings(
        llm=llm,
        base_embeddings=base_embeddings,
        template=template
    )

    vector_store = create_vector_store(documents, hyde_embeddings)

    regular_vector_store = create_vector_store(documents, base_embeddings)

    query = "What are chat loaders?"

    hyde_results = perform_similarity_search(vector_store, query)
    regular_results = perform_similarity_search(regular_vector_store, query)

    print_search_results(query, regular_results, "Standard Search")
    print_search_results(query, hyde_results, "HyDE Search")


def create_sample_documents() -> List[Document]:
    """
    Create sample documents for demonstration if loading fails.
    
    Returns:
        List[Document]: Sample documents
    """
    return [
        Document(page_content="LangSmith is a platform for LLM application development. "
                "It helps developers debug, evaluate, and enhance their LLM applications."),
        Document(page_content="Benchmarking question answering systems involves testing "
                "their ability to correctly answer questions from various sources including CSV data."),
        Document(page_content="Chat loaders are tools that help you finetune chat models using your own conversations. "
                "They convert chat data from various platforms into formats suitable for training language models. "
                "Using chat loaders, you can create AI systems that better match your communication style."),
        Document(page_content="Large language models (LLMs) represent a significant advancement in AI, "
                "particularly for natural language processing tasks.")
    ]


def main() -> None:
    """Execute the HyDE demonstration examples."""
    print("LangChain RAG Tips and Tricks: Hypothetical Document Embeddings (HyDE)")
    print("=" * 70)

    print("\nInitializing BGE embeddings...")
    try:
        base_embeddings = initialize_bge_embeddings(device="cpu")
    except Exception as e:
        print(f"Error initializing BGE embeddings: {e}")
        print("Please make sure sentence-transformers and related packages are installed.")
        return

    try:
        demo_basic_hyde(base_embeddings)

        demo_multi_generation_hyde(base_embeddings)

        demo_custom_prompt_hyde(base_embeddings)

        demo_hyde_for_retrieval(base_embeddings)
    
    except Exception as e:
        print(f"Error during demonstration: {e}")
    
    print("\n\nEnd of HyDE demonstrations.")


if __name__ == "__main__":
    main()
