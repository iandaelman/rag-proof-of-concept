from langchain_core.tools import tool, create_retriever_tool
from langgraph.graph import MessagesState

from app.retrieve.vector_store import init_vector_store
from app.utils.RetrievalMethod import RetrievalMethod

retriever = init_vector_store(RetrievalMethod.SIMILARITY_SEARCH)
retriever_tool = create_retriever_tool(
    retriever,
    "myminfin_support_retriever",
    "Search and return information about Myminfin support or contact information about ICT FOD Financiën. "
    "Use this tool for all non trivial questions asked."
)




@tool(response_format="content_and_artifact")
def myminfin_retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information about myminfin support questions.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i + 1}:\n{doc.page_content}")

    return "\n\n".join(results)



def should_retrieve(state: MessagesState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0