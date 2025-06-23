from langchain_core.messages import BaseMessage
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

from app.utils.RetrievalMethod import RetrievalMethod


class State(TypedDict):
    query: str
    context: List[Document]
    response: BaseMessage
    retrieval_method: RetrievalMethod
    persistent_directory:str
    store_name:str
    search_kwargs:dict
    model_name:str






