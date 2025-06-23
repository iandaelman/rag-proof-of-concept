# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

from langgraph.constants import START
from langgraph.graph import StateGraph

from app.generate import generate, generate_answer
from app.retriever import retrieve, retrieve_documents
from app.utils.RetrievalMethod import RetrievalMethod
from app.utils.State import State
from app.vector_store import init_vector_store

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    store_name = "chroma_db_doc"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    persistent_directory = os.path.join(base_dir, "vector_db", store_name)
    query = "Wie is Patrick Colmant"
    init_vector_store(document_path="knowledge-base-doc", db_name="chroma_db_doc")

    # relevant_docs = retrieve_documents(store_name="chroma_db_doc", query=query,
    #                                    retrieval_method=RetrievalMethod.SIMILARITY_SEARCH)
    # generate_answer(relevant_docs, query)

    state = graph.invoke(
        {"query": query,
         "retrieval_method": RetrievalMethod.SIMILARITY_SEARCH,
         "persistent_directory": persistent_directory,
         "store_name": store_name,
         "search_kwargs": {"k": 3},
         "model_name": "llama3.2"
         })
    print(f"\n--- Received response ---")
    print(state["response"])
    print(f"\n--- Received content ---")
    print(state["response"].content)

