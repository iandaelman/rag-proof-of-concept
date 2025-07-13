from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from app.generation.GenerateOrQuery import generate_query_or_respond
from app.augment.GradeDocuments import grade_documents
from app.augment.evalutation import evaluate_answers
from app.generation.generate import generate_answer
from app.retrieve.retriever import retriever_tool
from app.augment.rewrite_question import rewrite_question
from app.test_data import ragas_data_set

load_dotenv()
# response_model = get_response_model()

# TODO Vervangen door eigen retriever waar je meer controle over hebt


# response_model_with_tools = response_model.bind_tools([retriever_tool])
#
#



def main():
    # TODO create seperate states for the when the llm is stuck in a loop and maybe for treating the translations have been done
    workflow = StateGraph(MessagesState)

    # Define the nodes we will cycle between
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("tools", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

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
        grade_documents,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question"
        }

    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # Compile
    graph = workflow.compile()

    # user_input = input("Wat is uw vraag? ")
    # input_message = HumanMessage(content=user_input)

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
