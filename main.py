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
from app.retrieve.generate_or_query import generate_query_or_respond
from app.retrieve.retriever import myminfin_retriever_tool, retriever_tool, should_retrieve
from resources.test_data import ragas_data_set

load_dotenv()

def main():
    # TODO create seperate states for the when the llm is stuck in a loop and maybe for treating the translations have been done
    workflow = StateGraph(MessagesState)

    # Define the nodes we will cycle between
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    #workflow.add_node("tools", ToolNode([myminfin_retriever_tool]))
    workflow.add_node("tools", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer_with_evaluation)

    workflow.add_edge(START, "generate_query_or_respond")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        # Assess LLM decision (call `retriever_tool` tool or respond to the user)
        tools_condition
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "tools",
        # Assess agent decision
        grade_documents_with_evaluation,
        {
            "generate_answer_with_evaluation": "generate_answer_with_evaluation",
            "rewrite_question": "rewrite_question"
        }

    )
    workflow.add_edge("generate_answer_with_evaluation", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # Compile
    graph = workflow.compile()

    for question in ragas_data_set["user_input"]:
        print("Processing question:", question)
        input_message = HumanMessage(content=question)
        for chunk in graph.stream({"messages": [input_message]}, stream_mode="updates"):
            for node, update in chunk.items():
                print("Update from node", node)
                update["messages"][-1].pretty_print()
                print("\n\n")

    evaluate_answers()


if __name__ == '__main__':
    main()
