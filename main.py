# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import asyncio
import os

from app.generate import generate_answer
from app.retriever import retrieve_documents, RetrievalMethod
from app.vector_store import init_vector_store

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    query = "Wie is Patrick Colmant"
    document_source_extension = "doc"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    persistent_directory = os.path.join(base_dir, "vector_db", "chroma_db")
    db_name = f"chroma_db_{document_source_extension}"
    init_vector_store(document_path=f"knowledge-base-{document_source_extension}", db_name=db_name)
    relevant_docs = retrieve_documents(store_name=db_name, query=query,
                                       retrieval_method=RetrievalMethod.SIMILARITY_SEARCH)
    generate_answer(relevant_docs, query)
