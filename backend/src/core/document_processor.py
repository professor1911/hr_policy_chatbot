#document_processor.py

import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import glob
import concurrent.futures
import time
import logging
import re
from src.config.config import MODEL_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

# Global cache for embedding model to prevent reloading
_EMBEDDING_MODEL = None

def get_embedding_model():
    """Get a singleton embedding model to avoid reloading."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name=MODEL_CONFIG["embedding_model"]
        )
    return _EMBEDDING_MODEL

def load_document(file_path):
    """Load a document based on its file type."""
    try:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx') or file_path.endswith('.doc'):
            loader = UnstructuredFileLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return []
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []

def load_documents_parallel(file_list, max_workers=4):
    """Load documents in parallel using a thread pool."""
    documents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(load_document, file_path): file_path for file_path in file_list}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
            file_path = future_to_file[future]
            try:
                docs = future.result()
                documents.extend(docs)
                # Show progress every 10 files or at the end
                if (i + 1) % 10 == 0 or (i + 1) == len(file_list):
                    logger.info(f"Processed {i+1}/{len(file_list)} documents")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
    return documents

def process_documents(directory_path, progress_callback=None, status_callback=None): # Changed from progress_bar, status_text
    """Process all documents in a directory with status updates."""
    start_time = time.time()

    # Collect all valid files
    file_list = []
    for extension in ['.pdf', '.docx', '.doc', '.txt']:
        file_list.extend(glob.glob(os.path.join(directory_path, f'**/*{extension}'), recursive=True))

    if status_callback:
        status_callback(f"Found {len(file_list)} documents to process")

    if not file_list:
        logger.warning(f"No supported documents found in {directory_path}")
        return []

    # Determine optimal number of workers based on file count
    num_workers = min(4, len(file_list))

    # Process documents in parallel
    documents = load_documents_parallel(file_list, max_workers=num_workers)

    if progress_callback:
        progress_callback(1.0)

    if status_callback:
        status_callback(f"Processed {len(file_list)} documents in {time.time() - start_time:.2f} seconds")

    return documents

def is_policy_heading(text):
    """Check if a text chunk appears to be a policy heading or title."""
    # Look for patterns that indicate a heading/title
    patterns = [
        r'^.*policy.*$',
        r'^.*guideline.*$',
        r'^.*procedure.*$',
        r'^.*benefit.*$',
        r'^.*reimbursement.*$',
        r'^.*allowance.*$',
        r'^\s*[\d\.]+\s+[A-Z]',  # Numbered sections starting with capital letter
        r'^\s*[A-Z][A-Z\s]+:',   # ALL CAPS followed by colon
        r'^\s*[A-Z][a-z]+\s+[A-Z]'  # Title Case phrases
    ]

    # Check if any pattern matches the text (case insensitive for most patterns)
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False

def has_numerical_data(text):
    """Check if text contains important numerical data like monetary values."""
    # Patterns for various numerical data
    patterns = [
        r'INR\s+[\d,]+',  # Indian Rupees
        r'Rs\.?\s+[\d,]+',  # Rs. format
        r'\d+\s*%',  # Percentages
        r'\d+X',  # Multiplier format (like 2X)
        r'\d+\s*days',  # Days counts
        r'\d+\s*lacs',  # Lacs format
        r'\d+\.\d+',  # Decimal numbers
    ]

    # Check if any pattern matches
    for pattern in patterns:
        if re.search(pattern, text):
            return True

    return False

def split_documents(documents, chunk_size=800, chunk_overlap=150):
    """
    Split documents into smaller chunks with special handling for policy documents.
    This uses more intelligent chunk boundaries to preserve policy information.
    """
    logger.info(f"Splitting {len(documents)} documents into chunks with enhanced policy awareness")

    # Define custom separator for policy documents
    policy_separators = [
        # First try to split on major section boundaries
        "\n\n# ", "\n## ", "\n### ",
        "\n\nSection", "\n\nARTICLE",
        "\n\nPOLICY", "\n\nGUIDELINE",

        # Then try paragraph breaks
        "\n\n",

        # Then try line breaks
        "\n",

        # Then sentence boundaries
        ". ", "! ", "? ",

        # Finally character level
        " ", ""
    ]

    # Create a splitter that respects policy document structure
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=policy_separators,
        length_function=len
    )

    # Special handling for chunks with policy headings or numerical data
    all_chunks = []
    for doc in documents:
        # Get the basic chunks
        basic_chunks = text_splitter.split_documents([doc])

        # Propagate page number and source metadata to each chunk if present
        for chunk in basic_chunks:
            if 'page' in doc.metadata:
                # Store as 1-indexed for user-friendly display
                chunk.metadata['page'] = int(doc.metadata['page']) + 1
            if 'source' in doc.metadata:
                chunk.metadata['source'] = doc.metadata['source']

        # Process each chunk, possibly merging with the next chunk
        i = 0
        while i < len(basic_chunks) - 1:
            current_chunk = basic_chunks[i]
            next_chunk = basic_chunks[i+1]

            # Conditions for merging with next chunk:
            # 1. Current chunk has a heading but is very short
            # 2. Current chunk has heading and next chunk has important numerical data

            current_content = current_chunk.page_content
            next_content = next_chunk.page_content

            if is_policy_heading(current_content) and (
                len(current_content) < 200 or has_numerical_data(next_content)
            ):
                # Merge current and next chunks
                merged_chunk = current_chunk.copy()
                merged_chunk.page_content = current_content + "\n\n" + next_content
                # Propagate metadata for merged chunk
                merged_chunk.metadata = dict(current_chunk.metadata)
                all_chunks.append(merged_chunk)
                i += 2  # Skip both chunks since we've merged them
            else:
                all_chunks.append(current_chunk)
                i += 1

            # Handle the last chunk if we didn't merge it
            if i == len(basic_chunks) - 1:
                all_chunks.append(basic_chunks[-1])

    logger.info(f"Created {len(all_chunks)} chunks with enhanced policy awareness")
    return all_chunks

def create_vector_store(chunks, persist_directory="./hr_chroma_db"):
    """Create a vector store from document chunks."""
    logger.info(f"Creating vector store from {len(chunks)} chunks")
    start_time = time.time()

    # Use cached embedding model
    embeddings = get_embedding_model()

    # Create vector store with optimized batch size
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}  # Optimize for semantic search
    )
    vector_store.persist()

    logger.info(f"Vector store created in {time.time() - start_time:.2f} seconds")
    return vector_store

def run_document_processing(directory_path="./hr_documents"):
    """Run the entire document processing pipeline."""
    logger.info("### Document Processing ###")

    # Callbacks for logging progress/status instead of Streamlit components
    def log_progress(value):
        logger.info(f"Processing progress: {value*100:.1f}%")

    def log_status(text):
        logger.info(f"Processing status: {text}")

    # Process documents
    raw_documents = process_documents(directory_path, log_progress, log_status)

    if not raw_documents:
        logger.error("No documents were processed. Please check the directory and file formats.")
        return None

    logger.info(f"Loaded {len(raw_documents)} document sections")

    # Split documents
    logger.info("Splitting documents into chunks with enhanced policy awareness...")
    chunks = split_documents(raw_documents)

    logger.info(f"Created {len(chunks)} document chunks")

    # Create vector store
    logger.info("Creating vector embeddings (this may take a while)...")
    vector_store = create_vector_store(chunks)

    logger.info("Knowledge base created successfully!")
    return vector_store

def load_vector_store(persist_directory="./hr_chroma_db"):
    """Load an existing vector store."""
    start_time = time.time()
    logger.info(f"Loading vector store from {persist_directory}")

    try:
        # Use cached embedding model
        embeddings = get_embedding_model()

        # Load the vector store
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

        logger.info(f"Vector store loaded in {time.time() - start_time:.2f} seconds")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None

def estimate_tokens_for_context(docs, query):
    """Estimate token count for a query and its retrieved documents."""
    # Combine all text
    doc_texts = [doc.page_content for doc in docs]
    all_text = " ".join([query] + doc_texts)

    # Estimate tokens (approximately 4 chars per token)
    return len(all_text) // 4

def extract_section_name(text):
    """Extract section name from text."""
    # Look for patterns that indicate a heading/title
    patterns = [
        r'^.*policy.*$',
        r'^.*guideline.*$',
        r'^.*procedure.*$',
        r'^.*benefit.*$',
        r'^.*reimbursement.*$',
        r'^.*allowance.*$',
        r'^\s*[\d\.]+\s+[A-Z]',  # Numbered sections starting with capital letter
        r'^\s*[A-Z][A-Z\s]+:',   # ALL CAPS followed by colon
        r'^\s*[A-Z][a-z]+\s+[A-Z]'  # Title Case phrases
    ]

    # Check if any pattern matches the text (case insensitive for most patterns)
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return text.strip()

    return "Unknown Section"