import bootstrap  # noqa: F401 — must be first

import concurrent.futures
import glob
import logging
import os
import re
import time

from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.config import MODEL_CONFIG

logger = logging.getLogger(__name__)
_EMBEDDING_MODEL = None


def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name=MODEL_CONFIG["embedding_model"]
        )
    return _EMBEDDING_MODEL


def load_document(file_path):
    try:
        lower = file_path.lower()
        if lower.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif lower.endswith((".docx", ".doc")):
            loader = Docx2txtLoader(file_path)
        elif lower.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            logger.warning("Unsupported file type: %s", file_path)
            return []
        return loader.load()
    except Exception as exc:
        logger.error("Error loading %s: %s", file_path, exc)
        return []


def load_documents_parallel(file_list, max_workers=4):
    documents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(load_document, file_path): file_path
            for file_path in file_list
        }
        for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
            file_path = future_to_file[future]
            try:
                documents.extend(future.result())
                if (i + 1) % 10 == 0 or (i + 1) == len(file_list):
                    logger.info("Processed %s/%s documents", i + 1, len(file_list))
            except Exception as exc:
                logger.error("Error processing %s: %s", file_path, exc)
    return documents


def process_documents(directory_path, progress_callback=None, status_callback=None):
    start_time = time.time()
    file_list = []
    for extension in [".pdf", ".docx", ".doc", ".txt"]:
        file_list.extend(
            glob.glob(os.path.join(directory_path, f"**/*{extension}"), recursive=True)
        )

    if status_callback:
        status_callback(f"Found {len(file_list)} documents to process")

    if not file_list:
        logger.warning("No supported documents found in %s", directory_path)
        return []

    documents = load_documents_parallel(file_list, max_workers=min(4, len(file_list)))

    if progress_callback:
        progress_callback(1.0)
    if status_callback:
        status_callback(
            f"Processed {len(file_list)} documents in {time.time() - start_time:.2f} seconds"
        )
    return documents


def is_policy_heading(text):
    patterns = [
        r"^.*policy.*$",
        r"^.*guideline.*$",
        r"^.*procedure.*$",
        r"^.*benefit.*$",
        r"^.*reimbursement.*$",
        r"^.*allowance.*$",
        r"^\s*[\d\.]+\s+[A-Z]",
        r"^\s*[A-Z][A-Z\s]+:",
        r"^\s*[A-Z][a-z]+\s+[A-Z]",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def has_numerical_data(text):
    patterns = [
        r"INR\s+[\d,]+",
        r"Rs\.?\s+[\d,]+",
        r"\d+\s*%",
        r"\d+X",
        r"\d+\s*days",
        r"\d+\s*lacs",
        r"\d+\.\d+",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def split_documents(documents, chunk_size=800, chunk_overlap=150):
    logger.info("Splitting %s documents into chunks", len(documents))

    policy_separators = [
        "\n\n# ",
        "\n## ",
        "\n### ",
        "\n\nSection",
        "\n\nARTICLE",
        "\n\nPOLICY",
        "\n\nGUIDELINE",
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
        " ",
        "",
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=policy_separators,
        length_function=len,
    )

    all_chunks = []
    for doc in documents:
        basic_chunks = text_splitter.split_documents([doc])
        for chunk in basic_chunks:
            if "page" in doc.metadata:
                chunk.metadata["page"] = int(doc.metadata["page"]) + 1
            if "source" in doc.metadata:
                chunk.metadata["source"] = doc.metadata["source"]

        i = 0
        while i < len(basic_chunks) - 1:
            current_chunk = basic_chunks[i]
            next_chunk = basic_chunks[i + 1]
            current_content = current_chunk.page_content
            next_content = next_chunk.page_content

            if is_policy_heading(current_content) and (
                len(current_content) < 200 or has_numerical_data(next_content)
            ):
                merged_chunk = current_chunk.copy()
                merged_chunk.page_content = current_content + "\n\n" + next_content
                merged_chunk.metadata = dict(current_chunk.metadata)
                all_chunks.append(merged_chunk)
                i += 2
            else:
                all_chunks.append(current_chunk)
                i += 1

            if i == len(basic_chunks) - 1:
                all_chunks.append(basic_chunks[-1])

    logger.info("Created %s chunks", len(all_chunks))
    return all_chunks


def create_vector_store(chunks, persist_directory="./hr_chroma_db"):
    logger.info("Creating vector store from %s chunks", len(chunks))
    start_time = time.time()
    embeddings = get_embedding_model()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    logger.info("Vector store created in %.2f seconds", time.time() - start_time)
    return vector_store


def run_document_processing(directory_path="./hr_documents"):
    logger.info("### Document Processing ###")

    def log_progress(value):
        logger.info("Processing progress: %.1f%%", value * 100)

    def log_status(text):
        logger.info("Processing status: %s", text)

    raw_documents = process_documents(directory_path, log_progress, log_status)
    if not raw_documents:
        logger.error("No documents were processed.")
        return None

    chunks = split_documents(raw_documents)
    if not chunks:
        logger.error("No chunks were created.")
        return None

    return create_vector_store(chunks)


def load_vector_store(persist_directory="./hr_chroma_db"):
    start_time = time.time()
    logger.info("Loading vector store from %s", persist_directory)

    if not os.path.isdir(persist_directory):
        logger.info("Vector store directory does not exist yet.")
        return None

    try:
        embeddings = get_embedding_model()
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        try:
            if vector_store._collection.count() == 0:
                logger.info("Vector store exists but is empty.")
                return None
        except Exception:
            pass
        logger.info("Vector store loaded in %.2f seconds", time.time() - start_time)
        return vector_store
    except Exception as exc:
        logger.error("Error loading vector store: %s", exc)
        return None
