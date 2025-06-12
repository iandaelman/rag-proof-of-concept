import glob
import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader,
)
from langchain.text_splitter import CharacterTextSplitter

# File extensions to loader mapping
loader_mapping = {
    ".md": TextLoader,
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
}

def add_metadata(doc, doc_type):
    doc.metadata["doc_type"] = doc_type
    return doc

def load_documents_from_knowledge_base(path="knowledge-base"):
    documents = []
    folders = [f for f in glob.glob(f"{path}/*") if os.path.isdir(f)]

    for folder in folders:
        doc_type = os.path.basename(folder)

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

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)
