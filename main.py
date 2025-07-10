from pprint import pprint

from IPython.display import Image, display
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import create_retriever_tool
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from app.GenerateOrQuery import generate_query_or_respond
from app.GradeDocuments import grade_documents
from app.generate import generate_answer
from app.rewrite_question import rewrite_question
from app.vector_store import init_vector_store

load_dotenv()
# response_model = ChatOllama(model="granite3.3:8b", temperature=0)
response_model = init_chat_model(model="openai:gpt-4o-mini", temperature=0)
retriever = init_vector_store()
retriever_tool = create_retriever_tool(
    retriever,
    "myminfin_support_retriever",
    "Search and return information about Myminfin support or contact information about ICT FOD Financiën."
)

response_model_with_tools = response_model.bind_tools([retriever_tool])


def query_or_respond(state: MessagesState):
    """
    This methods will call the retriever tool when given a quetion about the MyMinfin support.
    In the case of a trivial question it will simply provide a response
    Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = response_model_with_tools.invoke(state["messages"][-1:])

    print("STATE" + state["messages"][-1].content)

    return {"messages": [response]}


def main():
    workflow = StateGraph(MessagesState)

    # Define the nodes we will cycle between
    workflow.add_node(query_or_respond)
    workflow.add_node("tools", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "query_or_respond")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "query_or_respond",
        # Assess LLM decision (call `retriever_tool` tool or respond to the user)
        tools_condition,
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "tools",
        # Assess agent decision
        grade_documents,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question"
        }

    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "query_or_respond")

    # Compile
    graph = workflow.compile()

    for chunk in graph.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Wie is Patrick Colmant?",
                    }
                ]
            }
    ):
        for node, update in chunk.items():
            print("Update from node", node)
            update["messages"][-1].pretty_print()
            print("\n\n")


if __name__ == '__main__':
    main()
