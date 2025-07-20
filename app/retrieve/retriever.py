from langchain_core.tools import tool

from app.retrieve.vector_store import init_vector_store
from app.utils.RetrievalMethod import RetrievalMethod

retriever = init_vector_store(RetrievalMethod.SIMILARITY_SEARCH)

@tool(response_format="content_and_artifact")
def myminfin_retriever_tool(query: str) -> tuple[str, list]:
    """
    This tool searches and returns the information about myminfin support questions.
    """
    print(query)
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information", []

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i + 1}:\n{doc.page_content}")

    summary_text = "\n\n".join(results)

    # You can optionally create a structured "artifact" (e.g., a list of dicts)
    artifact = [doc.page_content for doc in docs]

    return summary_text, artifact
