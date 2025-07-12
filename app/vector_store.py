import glob
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader)
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.chat_model import get_embedding


def add_metadata(doc, doc_type):
    doc.metadata["doc_type"] = doc_type
    return doc


def build_vector_store(embeddings_function, document_path, db_name):
    # Define the directory containing the text file and the persistent directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    persistent_directory = os.path.join(base_dir, "vector_db", db_name)
    loader_mapping = {
        ".md": TextLoader,
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
    }

    # Check if the Chroma vector store already exists
    print("Persistent directory does not exist. Initializing vector store...")

    documents = []
    # Iterate over each folder in the path
    for folder in glob.glob(f"{document_path}/*"):
        print(folder)
        if os.path.isdir(folder):
            # Iterate over each file in the folder
            for ext, loader_cls in loader_mapping.items():
                for doc_file_path in glob.glob(os.path.join(folder, f"*{ext}")):
                    print(doc_file_path)
                    try:
                        # Use the appropriate loader based on the file extension
                        loader = loader_cls(doc_file_path,encoding="utf-8")
                        doc = loader.load()
                        for d in doc:
                            d.metadata = {"source": doc_file_path, "folder": folder}

                            documents.append(d)
                    except Exception as e:
                        print(f"Failed to load file {doc_file_path}: {e}")

        # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")

    return Chroma.from_documents(
        docs, embeddings_function, persist_directory=persistent_directory).as_retriever()


def init_vector_store(document_path="knowledge-base-md", db_name="chroma_db_md") -> VectorStoreRetriever:
    embeddings_function = get_embedding()
    embedding_name = embeddings_function.__class__.__name__

    # Clean embedding name (e.g., Ollama)
    embedding_name_clean = embedding_name.replace("Embeddings", "")
    print("Embedding model being used:", embedding_name_clean)

    # Combine db_name with embedding name
    full_db_name = f"{db_name}_{embedding_name_clean}"

    # Construct the full persistent directory path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    persistent_directory = os.path.join(base_dir, "vector_db", full_db_name)

    if not os.path.exists(persistent_directory):
        return build_vector_store(
            embeddings_function=embeddings_function,
            document_path=document_path,
            db_name=full_db_name
        )
    else:
        print("Vector store already exists. No need to initialize.")
        return Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings_function
        ).as_retriever()
