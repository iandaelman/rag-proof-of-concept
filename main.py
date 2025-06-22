# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from app.retriever import test_different_retrieval_methods
from app.vector_store import init_vector_store

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    query = "Nationality Change"
    init_vector_store()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    test_different_retrieval_methods(query=query)
