from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState

from app.retrieve.retriever import myminfin_retriever_tool
from app.utils.configuration import get_response_model
from app.utils.prompts import RETRIEVE_DOCUMENTS_OR_RESPOND_PROMPT

load_dotenv()

response_model = get_response_model()

def retrieve_documents_or_respond(state: MessagesState) -> MessagesState:
    """
    This methods will call the retriever tool when given a non trivial question is asked.
    In the case of a trivial question it will simply provide a response
    Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    message = state["messages"][-1].content

    prompt = RETRIEVE_DOCUMENTS_OR_RESPOND_PROMPT.format(message=message)

    response_model_with_tools = response_model.bind_tools([myminfin_retriever_tool])
    response = response_model_with_tools.invoke([SystemMessage(content=prompt),
                                                 HumanMessage(content=message)])
    return MessagesState(messages=[response])

# Oude methode die niet werkte bij modellen met hogere parameters
# def retrieve_query_or_respond(state: MessagesState):
#     """
#     This methods will call the retriever tool when given a non trivial question is asked.
#     In the case of a trivial question it will simply provide a response
#     Call the model to generate a response based on the current state. Given
#     the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
#     """
#     response_model_with_tools = response_model.bind_tools([myminfin_retriever_tool])
#     response = response_model_with_tools.invoke(state["messages"][-1:])
#
#     return {"messages": [response]}
