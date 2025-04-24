"""
LangChain Advanced RAG Tips and Tricks 04 - Contextual Compression + Filtering

This script demonstrates how to enhance retrieval performance using:
1. Contextual compression with LLMs to extract relevant content
2. Filtering methods to remove irrelevant documents
3. Document transformation pipelines for advanced processing
"""

import warnings
import importlib
from typing import List, Any

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_openai import OpenAI

def dynamic_import(module_paths, class_name):
    """Try multiple import paths and return the first successful import."""
    for module_path in module_paths:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, class_name):
                return getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
    raise ImportError(f"Could not import {class_name} from any of the provided paths.")

try:
    from langchain_community.retrievers.contextual_compression import ContextualCompressionRetriever
except ImportError:
    try:
        from langchain.retrievers import ContextualCompressionRetriever
    except ImportError:
        print("ERROR: ContextualCompressionRetriever could not be imported.")
        raise

LLMChainExtractor = dynamic_import([
    'langchain_community.retrievers.document_compressors',
    'langchain.retrievers.document_compressors'
], 'LLMChainExtractor')

LLMChainFilter = dynamic_import([
    'langchain_community.retrievers.document_compressors',
    'langchain.retrievers.document_compressors'
], 'LLMChainFilter')

EmbeddingsFilter = dynamic_import([
    'langchain_community.retrievers.document_compressors',
    'langchain.retrievers.document_compressors'
], 'EmbeddingsFilter')

DocumentCompressorPipeline = dynamic_import([
    'langchain_community.retrievers.document_compressors',
    'langchain.retrievers.document_compressors'
], 'DocumentCompressorPipeline')

try:
    from langchain_community.document_transformers import EmbeddingsRedundantFilter
except ImportError:
    try:
        from langchain.document_transformers import EmbeddingsRedundantFilter
    except ImportError:
        print("Warning: EmbeddingsRedundantFilter could not be imported.")
        class EmbeddingsRedundantFilter:
            """Placeholder for EmbeddingsRedundantFilter if not available."""
            def __init__(self, *args, **kwargs):
                print("Warning: Using placeholder for EmbeddingsRedundantFilter")


def initialize_embeddings() -> Any:
    """
    Initialize and return embeddings model.
        
    Returns:
        Embeddings model instance
    """
    try:
        model_name = "BAAI/bge-small-en-v1.5"
        encode_kwargs = {'normalize_embeddings': True}

        return HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs=encode_kwargs
        )
    except Exception as e:
        print(f"Error initializing BGE embeddings: {e}")
        print("Falling back to OpenAI embeddings.")
        return OpenAIEmbeddings()


def load_and_split_documents(file_paths: List[str]) -> List[Document]:
    """
    Load documents from file paths and split them into chunks.
    
    Args:
        file_paths: List of file paths to load
        
    Returns:
        List of document chunks
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)


def create_base_retriever(documents: List[Document], embeddings: Any, k: int = 4) -> BaseRetriever:
    """
    Create a base retriever from documents and embeddings.
    
    Args:
        documents: List of documents to index
        embeddings: Embeddings model to use
        k: Number of documents to retrieve
        
    Returns:
        BaseRetriever: Configured retriever
    """
    return FAISS.from_documents(
        documents, 
        embeddings
    ).as_retriever(search_kwargs={"k": k})


def create_llm_chain_extractor(temperature: float = 0) -> Any:
    """
    Create an LLM-based document content extractor.
    
    This extractor uses an LLM to extract only the relevant parts of documents
    that answer a specific query.
    
    Args:
        temperature: LLM temperature parameter
        
    Returns:
        LLMChainExtractor: Configured extractor
    """
    llm = OpenAI(temperature=temperature)
    return LLMChainExtractor.from_llm(llm)


def create_llm_chain_filter(temperature: float = 0) -> Any:
    """
    Create an LLM-based document filter.
    
    This filter uses an LLM to determine if a document is relevant to a query
    with a simple YES/NO decision.
    
    Args:
        temperature: LLM temperature parameter
        
    Returns:
        LLMChainFilter: Configured filter
    """
    llm = OpenAI(temperature=temperature)
    return LLMChainFilter.from_llm(llm)


def create_embeddings_filter(
    embeddings: Any, 
    similarity_threshold: float = 0.76
) -> Any:
    """
    Create an embeddings-based document filter.
    
    This filter removes documents that have an embedding similarity
    below the specified threshold compared to the query.
    
    Args:
        embeddings: Embeddings model to use
        similarity_threshold: Minimum similarity score (0-1)
        
    Returns:
        EmbeddingsFilter: Configured filter
    """
    return EmbeddingsFilter(
        embeddings=embeddings, 
        similarity_threshold=similarity_threshold
    )


def create_document_compressor_pipeline(
    transformers: List[Any]
) -> Any:
    """
    Create a pipeline of document transformers and compressors.
    
    Args:
        transformers: List of document transformers and compressors
        
    Returns:
        DocumentCompressorPipeline: Configured pipeline
    """
    return DocumentCompressorPipeline(transformers=transformers)


def create_compression_retriever(
    base_retriever: BaseRetriever,
    base_compressor: Any
) -> Any:
    """
    Create a retriever that applies contextual compression.
    
    Args:
        base_retriever: Base retriever to use
        base_compressor: Compressor to apply to retrieved documents
        
    Returns:
        ContextualCompressionRetriever: Configured retrieval system
    """
    return ContextualCompressionRetriever(
        base_compressor=base_compressor,
        base_retriever=base_retriever
    )


def pretty_print_docs(docs: List[Document]) -> None:
    """
    Print documents in a formatted way.
    
    Args:
        docs: List of documents to print
    """
    if not docs:
        print("No documents to display.")
        return
        
    print(f"\n{'-' * 100}\n".join([
        f"Document {i+1}:\n\n" + d.page_content 
        for i, d in enumerate(docs)
    ]))


def retrieval_example_llm_extractor(
    retriever: BaseRetriever,
    query: str,
    temperature: float = 0
) -> None:
    """
    Demonstrate retrieval with LLM-based content extraction.
    
    Args:
        retriever: Base retriever to use
        query: Query to run
        temperature: LLM temperature parameter
    """
    print("\n\n===== Example: LLM Chain Extractor =====")
    print(f"Query: '{query}'")
    
    try:
        base_docs = retriever.invoke(query)
        
        print("\nBase Retriever Results:")
        pretty_print_docs(base_docs)

        compressor = create_llm_chain_extractor(temperature)
        compression_retriever = create_compression_retriever(retriever, compressor)
        
        compressed_docs = compression_retriever.invoke(query)
        
        print("\nCompressed Results (LLM Extraction):")
        pretty_print_docs(compressed_docs)
    except Exception as e:
        print(f"Error in LLM extractor example: {e}")


def retrieval_example_llm_filter(
    retriever: BaseRetriever,
    query: str,
    temperature: float = 0
) -> None:
    """
    Demonstrate retrieval with LLM-based document filtering.
    
    Args:
        retriever: Base retriever to use
        query: Query to run
        temperature: LLM temperature parameter
    """
    print("\n\n===== Example: LLM Chain Filter =====")
    print(f"Query: '{query}'")
    
    try:
        llm_filter = create_llm_chain_filter(temperature)
        compression_retriever = create_compression_retriever(retriever, llm_filter)
        
        compressed_docs = compression_retriever.invoke(query)
        
        print("\nFiltered Results (LLM YES/NO Decision):")
        pretty_print_docs(compressed_docs)
    except Exception as e:
        print(f"Error in LLM filter example: {e}")


def retrieval_example_embeddings_filter(
    retriever: BaseRetriever,
    query: str,
    embeddings: Any,
    similarity_threshold: float = 0.76
) -> None:
    """
    Demonstrate retrieval with embeddings-based document filtering.
    
    Args:
        retriever: Base retriever to use
        query: Query to run
        embeddings: Embeddings model to use
        similarity_threshold: Minimum similarity score (0-1)
    """
    print("\n\n===== Example: Embeddings Filter =====")
    print(f"Query: '{query}'")
    print(f"Similarity Threshold: {similarity_threshold}")
    
    try:
        embeddings_filter = create_embeddings_filter(embeddings, similarity_threshold)
        compression_retriever = create_compression_retriever(retriever, embeddings_filter)
        
        compressed_docs = compression_retriever.invoke(query)
        
        print("\nFiltered Results (Embedding Similarity):")
        pretty_print_docs(compressed_docs)
    except Exception as e:
        print(f"Error in embeddings filter example: {e}")


def retrieval_example_simple_pipeline(
    retriever: BaseRetriever,
    query: str,
    embeddings: Any,
    similarity_threshold: float = 0.76
) -> None:
    """
    Demonstrate retrieval with a simple document transformation pipeline.
    
    This pipeline:
    1. Splits documents into smaller chunks
    2. Removes redundant chunks using embeddings
    3. Filters chunks by similarity to query
    
    Args:
        retriever: Base retriever to use
        query: Query to run
        embeddings: Embeddings model to use
        similarity_threshold: Minimum similarity score (0-1)
    """
    print("\n\n===== Example: Simple Pipeline =====")
    print(f"Query: '{query}'")
    
    try:
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = create_embeddings_filter(embeddings, similarity_threshold)

        pipeline_compressor = create_document_compressor_pipeline([
            splitter, 
            redundant_filter, 
            relevant_filter
        ])

        compression_retriever = create_compression_retriever(retriever, pipeline_compressor)
        compressed_docs = compression_retriever.invoke(query)
        
        print("\nPipeline Results (Split + Deduplicate + Filter):")
        pretty_print_docs(compressed_docs)
    except Exception as e:
        print(f"Error in simple pipeline example: {e}")


def retrieval_example_advanced_pipeline(
    retriever: BaseRetriever,
    query: str,
    embeddings: Any,
    temperature: float = 0,
    similarity_threshold: float = 0.76
) -> None:
    """
    Demonstrate retrieval with an advanced document transformation pipeline.
    
    This pipeline:
    1. Splits documents into smaller chunks
    2. Uses an LLM to extract relevant content from each chunk
    3. Removes redundant chunks using embeddings
    4. Filters chunks by similarity to query
    
    Args:
        retriever: Base retriever to use
        query: Query to run
        embeddings: Embeddings model to use
        temperature: LLM temperature parameter
        similarity_threshold: Minimum similarity score (0-1)
    """
    print("\n\n===== Example: Advanced Pipeline =====")
    print(f"Query: '{query}'")
    
    try:
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
        llm_compressor = create_llm_chain_extractor(temperature)
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = create_embeddings_filter(embeddings, similarity_threshold)

        pipeline_compressor = create_document_compressor_pipeline([
            splitter, 
            llm_compressor,
            redundant_filter, 
            relevant_filter
        ])

        compression_retriever = create_compression_retriever(retriever, pipeline_compressor)
        compressed_docs = compression_retriever.invoke(query)
        
        print("\nAdvanced Pipeline Results (Split + Extract + Deduplicate + Filter):")
        pretty_print_docs(compressed_docs)
    except Exception as e:
        print(f"Error in advanced pipeline example: {e}")


def main() -> None:
    """Execute the contextual compression and filtering examples."""
    print("LangChain RAG Tips and Tricks: Contextual Compression + Filtering")
    print("=" * 70)

    print("\nInitializing embeddings...")
    embeddings = initialize_embeddings()

    print("\nLoading and processing documents...")
    try:
        docs = load_and_split_documents([
            '/blog_posts/blog.langchain.dev_announcing-langsmith_.txt',
            '/blog_posts/blog.langchain.dev_benchmarking-question-answering-over-csv-data_.txt',
        ])
        
        if not docs:
            print("No documents loaded from files. Creating sample documents...")
            docs = create_sample_documents()
    except Exception as e:
        print(f"Error loading documents: {e}")
        print("Creating sample documents instead...")
        docs = create_sample_documents()
    
    print(f"Working with {len(docs)} document chunks.")

    print("\nCreating base retriever...")
    retriever = create_base_retriever(docs, embeddings)

    query = "What is LangSmith?"

    try:
        retrieval_example_llm_extractor(retriever, query)

        retrieval_example_llm_filter(retriever, query)

        retrieval_example_embeddings_filter(retriever, query, embeddings)

        retrieval_example_simple_pipeline(retriever, query, embeddings)

        retrieval_example_advanced_pipeline(retriever, query, embeddings)
    except Exception as e:
        print(f"Error running examples: {e}")
    
    print("\n\nEnd of examples.")


def create_sample_documents() -> List[Document]:
    """
    Create sample documents for demonstration if loading fails.
    
    Returns:
        List[Document]: Sample documents
    """
    return [
        Document(page_content="LangSmith is a platform for LLM application development, monitoring, and testing. "
                "It helps developers debug, monitor, evaluate, and improve their language model applications."),
        Document(page_content="LangSmith integrates with the LangChain framework to provide detailed tracing and debugging. "
                "It captures inputs, outputs, and intermediate steps of your LLM applications."),
        Document(page_content="LangChain is an open-source framework for building applications with large language models. "
                "It provides modules for chains, agents, memory, and more."),
        Document(page_content="Benchmarking question answering over CSV data is important for evaluating retrieval performance. "
                "This process involves measuring accuracy, latency, and relevance of responses.")
    ]


if __name__ == "__main__":
    main()
