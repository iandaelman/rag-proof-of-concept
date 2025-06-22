# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from app.generate import generate_answer
from app.retriever import retrieve_documents, RetrievalMethod
from app.vector_store import init_vector_store

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    query = "Who is Patrick Colmant"
    init_vector_store()
    relevant_docs = retrieve_documents(query="example query", retrieval_method=RetrievalMethod.SIMILARITY_SEARCH)
    generate_answer(relevant_docs)
