# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from langgraph.constants import START
from langgraph.graph import StateGraph

from app.generate import generate
from app.retriever import retrieve
from app.utils.State import State
from app.vector_store import init_vector_store

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    query = "Wie is Patrick Colmant"
    init_vector_store(document_path="knowledge-base-doc", db_name="chroma_db_doc")

    # relevant_docs = retrieve_documents(store_name="chroma_db_doc", query=query,
    #                                    retrieval_method=RetrievalMethod.SIMILARITY_SEARCH)
    response = graph.invoke({"query": query})
    print(response.answer)
