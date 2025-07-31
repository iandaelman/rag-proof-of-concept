import os

from langchain_core.tools import tool

from app.retrieve.vector_store import init_vector_store
from app.utils.RetrievalMethod import RetrievalMethod

retriever = init_vector_store(RetrievalMethod.SIMILARITY_SEARCH)

@tool(response_format="content_and_artifact")
def myminfin_retriever_tool(query: str) -> tuple[str, list]:
    """
    This tool searches and returns the information about myminfin support questions.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information", []

    results = []

    for doc in docs:
        source_path = doc.metadata.get("source", "Unknown source")
        source_file = os.path.basename(source_path) if source_path else "Unknown file"
        source_name = os.path.splitext(source_file)[0]
        results.append(
            f"Source: {source_name}\n{doc.page_content}"
        )
    summary_text = "\n\n".join(results)
    artifact = [doc.page_content for doc in docs]
    return summary_text, artifact
