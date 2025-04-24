"""
LangChain RAG Tips and Tricks 06 - RAGFusion with OpenAI and BGE Embeddings

This script demonstrates the RAGFusion technique which:
1. Generates multiple search queries from a single user query
2. Retrieves documents for each generated query
3. Combines results using Reciprocal Rank Fusion to improve retrieval quality
"""

import warnings
import os
import json
import textwrap
import zipfile
import requests
from io import BytesIO
from typing import List, Any, Tuple, Optional, Callable

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_openai import ChatOpenAI


def download_and_extract_zip(url: str, target_folder: str) -> None:
    """
    Download and extract a zip file from a URL.
    
    Args:
        url: URL to download the zip file from
        target_folder: Folder to extract the contents to
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {url}")

    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(target_folder)

    print(f"Files extracted to {target_folder}")


def zip_folder(folder_path: str, zip_file_path: str) -> None:
    """
    Compress a folder into a zip file.
    
    Args:
        folder_path: Path to the folder to compress
        zip_file_path: Path for the output zip file
    """
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                relative_path = os.path.relpath(os.path.join(root, file), os.path.dirname(folder_path))
                zipf.write(os.path.join(root, file), arcname=relative_path)

    print(f"{zip_file_path} created successfully.")


def wrap_text(text: str, width: int = 90) -> str:
    """
    Wrap text to a specified width while preserving newlines.
    
    Args:
        text: Text to wrap
        width: Maximum width for each line
        
    Returns:
        Wrapped text
    """
    lines = text.split('\n')

    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def download_sample_data() -> None:
    """
    Download sample Singapore text files and pre-built Chroma database.
    """
    text_url = "https://www.dropbox.com/scl/fi/av3nw07o5mo29cjokyp41/singapore_text_files_languages.zip?rlkey=xqdy5f1modtbnrzzga9024jyw&dl=1"
    chroma_url = "https://www.dropbox.com/scl/fi/3kep8mo77h642kvpum2p7/singapore_chroma_db.zip?rlkey=4ry4rtmeqdcixjzxobtmaajzo&dl=1"
    
    try:
        download_and_extract_zip(text_url, "singapore_text")
        print("Downloaded sample text files successfully.")

        download_and_extract_zip(chroma_url, ".")
        print("Downloaded pre-built Chroma database successfully.")
    except Exception as e:
        print(f"Error downloading sample data: {e}")
        print("Continuing without sample data.")


def load_documents_from_directory(
    directory_path: str,
    glob_pattern: str = "*.txt"
) -> List[Document]:
    """
    Load documents from a directory with a glob pattern.
    
    Args:
        directory_path: Path to the directory containing the documents
        glob_pattern: Glob pattern to match files
        
    Returns:
        List of loaded documents
    """
    try:
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_cls=TextLoader,
            show_progress=True
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from {directory_path}.")
        return documents
    except Exception as e:
        print(f"Error loading documents from {directory_path}: {e}")
        print("Using sample documents instead.")
        return create_sample_documents()


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[str]:
    """
    Split documents into chunks for better processing.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    raw_text = ''
    for doc in documents:
        text = doc.page_content
        if text:
            raw_text += text

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(raw_text)
    print(f"Split documents into {len(chunks)} chunks.")
    
    return chunks


def initialize_bge_embeddings(
    model_name: str = "BAAI/bge-small-en-v1.5",
    device: str = "cpu"
) -> HuggingFaceBgeEmbeddings:
    """
    Initialize BGE embeddings model.
    
    Args:
        model_name: Name of the BGE model to use
        device: Device to run the model on ('cpu' or 'cuda')
        
    Returns:
        Configured BGE embeddings model
    """
    encode_kwargs = {'normalize_embeddings': True}  # For cosine similarity
    
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs=encode_kwargs,
    )


def create_or_load_vector_db(
    texts: Optional[List[str]] = None,
    embedding_function: Any = None,
    persist_directory: str = "./chroma_db"
) -> Chroma:
    """
    Create a new vector database or load an existing one.
    
    Args:
        texts: Text chunks to index (if creating a new DB)
        embedding_function: Embedding function to use
        persist_directory: Directory for persistence
        
    Returns:
        Chroma vector database
    """
    if texts and embedding_function and not os.path.exists(persist_directory):
        print(f"Creating new vector store at {persist_directory}")
        db = Chroma.from_texts(
            texts,
            embedding_function,
            persist_directory=persist_directory
        )
        db.persist()
        return db
    else:
        print(f"Loading existing vector store from {persist_directory}")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )


def create_sample_documents() -> List[Document]:
    """
    Create sample documents about Singapore if loading real documents fails.
    
    Returns:
        List of sample documents
    """
    return [
        Document(page_content="Singapore, officially the Republic of Singapore, is a sovereign island city-state in maritime Southeast Asia."),
        Document(page_content="Universal Studios Singapore is a theme park located within Resorts World Sentosa on Sentosa Island, Singapore. It features 28 rides, shows, and attractions in seven themed zones."),
        Document(page_content="The Singapore Zoo, formerly known as the Singapore Zoological Gardens and commonly known locally as the Mandai Zoo, occupies 28 hectares on the margins of Upper Seletar Reservoir within Singapore's heavily forested central catchment area."),
        Document(page_content="Marina Bay Sands is an integrated resort fronting Marina Bay in Singapore, owned by the Las Vegas Sands corporation. It is the world's most expensive standalone casino property at US$8 billion."),
        Document(page_content="Sentosa is an island resort off Singapore's southern coast. It can be reached by road, cable car, pedestrian boardwalk and monorail.")
    ]


def setup_query_generator(temperature: float = 0) -> Callable:
    """
    Set up a chain that generates multiple search queries from a single query.
    
    Args:
        temperature: LLM temperature parameter
        
    Returns:
        Callable chain function that generates multiple queries
    """
    prompt = ChatPromptTemplate(
        input_variables=['question'],
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a helpful assistant that generates multiple search queries based on a single input query."
            ),
            HumanMessagePromptTemplate.from_template(
                "Generate multiple search queries related to: {question} \n OUTPUT (4 queries):"
            )
        ]
    )

    return (
        prompt 
        | ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature) 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )


def dumps(obj: Any) -> str:
    """
    Serialize an object to a JSON string for deduplication.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string representation
    """
    if isinstance(obj, Document):
        return json.dumps({"page_content": obj.page_content, "metadata": obj.metadata})
    return json.dumps(obj)


def loads(json_str: str) -> Any:
    """
    Deserialize a JSON string back to an object.
    
    Args:
        json_str: JSON string to deserialize
        
    Returns:
        Deserialized object
    """
    data = json.loads(json_str)
    if isinstance(data, dict) and "page_content" in data:
        return Document(page_content=data["page_content"], metadata=data.get("metadata", {}))
    return data


def reciprocal_rank_fusion(results: List[List[Document]], k: int = 60) -> List[Tuple[Document, float]]:
    """
    Implement Reciprocal Rank Fusion algorithm to combine results from multiple retrievers.
    
    RRF gives each document a score = sum(1 / (rank + k)) across all retrievers.
    
    Args:
        results: List of document lists from different retrievers
        k: Constant to avoid division by zero and control influence of high rankings
        
    Returns:
        List of (document, score) tuples sorted by score
    """
    fused_scores = {}
    
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    return [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]


def setup_standard_rag_chain(
    retriever: Any,
    temperature: float = 0
) -> Callable:
    """
    Set up a standard RAG chain with a single retriever.
    
    Args:
        retriever: Document retriever to use
        temperature: LLM temperature parameter
        
    Returns:
        Callable chain function
    """
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )


def setup_ragfusion_chain(
    retriever: Any,
    query_generator: Callable,
    temperature: float = 0
) -> Callable:
    """
    Set up a RAGFusion chain that uses multiple generated queries.
    
    Args:
        retriever: Document retriever to use
        query_generator: Function that generates multiple queries
        temperature: LLM temperature parameter
        
    Returns:
        Callable chain function
    """
    ragfusion_retriever = query_generator | retriever.map() | reciprocal_rank_fusion

    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)

    return (
        {
            "context": ragfusion_retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )


def format_ragfusion_results(
    results: List[Tuple[Document, float]]
) -> str:
    """
    Format RAGFusion results for display.
    
    Args:
        results: List of (document, score) tuples
        
    Returns:
        Formatted string representation
    """
    output = "\nRAGFusion Results:\n" + "=" * 40 + "\n"
    
    for i, (doc, score) in enumerate(results):
        output += f"\nResult {i+1} (Score: {score:.4f}):\n"
        output += f"{doc.page_content[:150]}...\n"
        
    return output


def compare_standard_vs_ragfusion(
    query: str,
    standard_chain: Callable,
    ragfusion_chain: Callable
) -> None:
    """
    Compare results from standard RAG and RAGFusion approaches.
    
    Args:
        query: Query string to use for comparison
        standard_chain: Standard RAG chain function
        ragfusion_chain: RAGFusion chain function
    """
    print(f"\n\n===== Comparing Standard RAG vs RAGFusion for Query: '{query}' =====")

    try:
        print("\nRunning standard RAG chain...")
        standard_result = standard_chain.invoke(query)
        
        print("\nRunning RAGFusion chain...")
        ragfusion_result = ragfusion_chain.invoke({"question": query})

        print("\n\n===== RESULTS =====")
        print("\nStandard RAG Result:")
        print("-" * 40)
        print(wrap_text(standard_result))
        
        print("\n\nRAGFusion Result:")
        print("-" * 40)
        print(wrap_text(ragfusion_result))
        
    except Exception as e:
        print(f"Error during comparison: {e}")


def main() -> None:
    """Execute the RAGFusion with OpenAI and BGE Embeddings tutorial."""
    print("LangChain RAG Tips and Tricks: RAGFusion with OpenAI and BGE Embeddings")
    print("=" * 80)

    try:
        print("\nChecking for sample data...")
        if not os.path.exists("singapore_text") or not os.path.exists("./chroma_db"):
            print("Sample data not found. Downloading...")
            download_sample_data()
        else:
            print("Sample data already exists.")
    except Exception as e:
        print(f"Error with sample data: {e}")
        print("Continuing with existing data or samples.")

    print("\nInitializing BGE embeddings...")
    try:
        embedding_function = initialize_bge_embeddings(device="cpu")
    except Exception as e:
        print(f"Error initializing BGE embeddings: {e}")
        print("Please make sure sentence-transformers is installed.")
        return

    print("\nSetting up vector database...")
    try:
        db = create_or_load_vector_db(
            embedding_function=embedding_function,
            persist_directory="./chroma_db"
        )
    except Exception as e:
        print(f"Error setting up vector database: {e}")
        print("Please check if the Chroma database is accessible.")
        return

    print("\nSetting up retrieval chains...")
    retriever = db.as_retriever(search_kwargs={"k": 5})

    query_generator = setup_query_generator()

    standard_chain = setup_standard_rag_chain(retriever)

    ragfusion_chain = setup_ragfusion_chain(retriever, query_generator)

    print("\nComparing standard RAG vs RAGFusion...")
    sample_queries = [
        "Tell me about Universal Studios Singapore",
        "What are the best attractions in Singapore?",
        "Where can I find good food in Singapore?"
    ]
    
    for query in sample_queries:
        compare_standard_vs_ragfusion(query, standard_chain, ragfusion_chain)
    
    print("\n\nEnd of RAGFusion demonstration.")


if __name__ == "__main__":
    main()