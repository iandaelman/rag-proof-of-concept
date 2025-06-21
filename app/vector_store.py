import glob
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def add_metadata(doc, doc_type):
    doc.metadata["doc_type"] = doc_type
    return doc


def build_vector_store(path="knowledge-base", embeddings_function=None):
    # Define the directory containing the text file and the persistent directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    persistent_directory = os.path.join(base_dir, "vector_db", "chroma_db")
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
    for folder in glob.glob(f"{path}/*"):
        print(folder)
        if os.path.isdir(folder):
            # Iterate over each file in the folder
            for ext, loader_cls in loader_mapping.items():
                for doc_file_path in glob.glob(os.path.join(folder, f"*{ext}")):
                    print(doc_file_path)
                    try:
                        # Use the appropriate loader based on the file extension
                        loader = loader_cls(doc_file_path)
                        doc = loader.load()
                        for d in doc:
                            d.metadata = {"source": doc_file_path, "folder": folder}

                            documents.append(d)
                    except Exception as e:
                        print(f"Failed to load file {doc_file_path}: {e}")

        # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings_function, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")
    return db


def retrieve_vector_store():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    persistent_directory = os.path.join(base_dir, "vector_db", "chroma_db")
    print("\n--- Creating embeddings ---")
    embeddings_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("\n--- Finished creating embeddings ---")
    if not os.path.exists(persistent_directory):
        return build_vector_store(embeddings_function=embeddings_function)
    else:
        print("Vector store already exists. No need to initialize.")
        return Chroma(embedding_function=embeddings_function, persist_directory=persistent_directory)
