from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import create_retriever_tool
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from app.GradeDocuments import grade_documents
from app.chat_model import get_response_model
from app.generate import generate_answer
from app.rewrite_question import rewrite_question
from app.vector_store import init_vector_store

load_dotenv()
response_model = get_response_model()
#response_model = get_open_ai_model()
retriever = init_vector_store()
# TODO Vervangen door eigen retriever waar je meer controle over hebt
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

    return {"messages": [response]}


def main():
    # TODO create seperate states for the when the llm is stuck in a loop and maybe for treating the translations have been done
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

    # user_input = input("Wat is uw vraag? ")
    # input_message = HumanMessage(content=user_input)

    input_message = HumanMessage(content="Kan je me de contactgegevens van Patrick Colmant geven?")

    for chunk in graph.stream({"messages": [input_message]}, stream_mode="updates"):
        for node, update in chunk.items():
            print("Update from node", node)
            update["messages"][-1].pretty_print()
            print("\n\n")


if __name__ == '__main__':
    main()
