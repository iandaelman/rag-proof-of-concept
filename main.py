from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from app.augment.evalutation import evaluate_answers
from app.augment.grade_documents import grade_documents_with_evaluation
from app.augment.rewrite_question import rewrite_question
from app.generation.generate import generate_answer_with_evaluation
from app.retrieve.retrieve_or_respond import retrieve_query_or_respond, retrieve_query_or_respond_without_tool
from app.retrieve.retriever import myminfin_retriever_tool, myminfin_retriever
from resources.test_data import ragas_data_set

load_dotenv()


def main():
    workflow = StateGraph(MessagesState)

    # Define the nodes we will cycle between
    # workflow.add_node("retrieve_query_or_respond", retrieve_query_or_respond)
    # workflow.add_node("retrieve_query_or_respond_without_tool", retrieve_query_or_respond_without_tool)
    # workflow.add_node("tools", ToolNode([myminfin_retriever_tool]))
    workflow.add_node("myminfin_retriever", myminfin_retriever)
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer_with_evaluation)

    # workflow.add_edge(START, "retrieve_query_or_respond_without_tool")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        START,
        retrieve_query_or_respond_without_tool,
        # Assess LLM decision (call `retriever_tool` tool or respond to the user)
        {
            "myminfin_retriever": "myminfin_retriever",
            "__end__": "__end__"
        }
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "myminfin_retriever",
        # Assess agent decision
        grade_documents_with_evaluation,
        {
            "generate_answer_with_evaluation": "generate_answer_with_evaluation",
            "rewrite_question": "rewrite_question"
        }

    )
    workflow.add_edge("generate_answer_with_evaluation", END)
    workflow.add_edge("rewrite_question", "myminfin_retriever")

    # Compile
    graph = workflow.compile()

    for question in ragas_data_set["user_input"]:
        print("Processing question:", question)
        input_message = HumanMessage(content=question)
        for chunk in graph.stream({"messages": [input_message]}, stream_mode="updates"):
            for node, update in chunk.items():
                print("Update from node", node)
                print(update["messages"][-1])
                print("\n\n")

    evaluate_answers()


if __name__ == '__main__':
    main()
