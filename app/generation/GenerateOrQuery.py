from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState

from app.retrieve.retriever import retriever_tool
from app.utils.configuration import get_response_model

load_dotenv()

response_model = get_response_model()

QUERY_OR_RESPOND_PROMPT = """
This method decides whether to call the retriever tool or respond directly.

If the user's question is trivial, respond directly.  
If the question is non-trivial, use the retriever tool to generate a response.

Given the user's question:  
"{question}"

Determine whether the question is trivial. 
"""


def generate_query_or_respond(state: MessagesState) -> MessagesState:
    """
    This methods will call the retriever tool when given a non trivial question is asked.
    In the case of a trivial question it will simply provide a response
    Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    question = state["messages"][0].content

    prompt = QUERY_OR_RESPOND_PROMPT.format(question=question)

    response_model_with_tools = response_model.bind_tools([retriever_tool])
    response = response_model_with_tools.invoke([SystemMessage(content=prompt)])

    return MessagesState(messages=[response])

# Oude methode die niet werkte bij modellen met hogere parameters
# def generate_query_or_respond(state: MessagesState):
#     """
#     This methods will call the retriever tool when given a non trivial question is asked.
#     In the case of a trivial question it will simply provide a response
#     Call the model to generate a response based on the current state. Given
#     the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
#     """
#     response_model_with_tools = response_model.bind_tools([retriever_tool])
#     response = response_model_with_tools.invoke(state["messages"][-1:])
#
#     return {"messages": [response]}
