from langchain_core.tools import create_retriever_tool

from app.utils.RetrievalMethod import RetrievalMethod
from app.retrieve.vector_store import init_vector_store

retriever = init_vector_store(RetrievalMethod.SIMILARITY_SEARCH)
retriever_tool = create_retriever_tool(
    retriever,
    "myminfin_support_retriever",
    "Search and return information about Myminfin support or contact information about ICT FOD Financiën. "
    "Use this tool for all non trivial questions asked."
)