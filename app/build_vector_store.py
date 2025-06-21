import glob
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def add_metadata(doc, doc_type):
    doc.metadata["doc_type"] = doc_type
    return doc


def build_vector_store(path="knowledge-base"):
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
    if not os.path.exists(persistent_directory):
        print("Persistent directory does not exist. Initializing vector store...")

        documents = []
        folders = [f for f in glob.glob(f"{path}/*") if os.path.isdir(f)]
        # Ensure the text file exists
        for folder in folders:
            doc_type = os.path.basename(folder)
            # Read the text content from the file
            for ext, loader_cls in loader_mapping.items():
                loader = DirectoryLoader(
                    folder,
                    glob=f"**/*{ext}",
                    loader_cls=loader_cls,
                    loader_kwargs={'encoding': 'utf-8'} if loader_cls == TextLoader else {}
                )
                try:
                    docs = loader.load()
                    documents.extend([add_metadata(doc, doc_type) for doc in docs])
                except Exception as e:
                    print(f"Failed to load files from {folder} with extension {ext}: {e}")

            # Split the document into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)

            # Display information about the split documents
            print("\n--- Document Chunks Information ---")
            print(f"Number of document chunks: {len(docs)}")
            print(f"Sample chunk:\n{docs[0].page_content}\n")

            # Create embeddings
            print("\n--- Creating embeddings ---")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            # Update to a valid embedding model if needed
            print("\n--- Finished creating embeddings ---")

            # Create the vector store and persist it automatically
            print("\n--- Creating vector store ---")
            Chroma.from_documents(
                docs, embeddings, persist_directory=persistent_directory)
            print("\n--- Finished creating vector store ---")
    else:
        print("Vector store already exists. No need to initialize.")
