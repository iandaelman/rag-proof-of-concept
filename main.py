# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from langgraph.graph import END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from app.generate import generate_tool_chain
from app.retriever import retrieve_tool_chain, query_or_respond


# Press the green button in the gutter to run the script.

def main():
    tools = ToolNode([retrieve_tool_chain])
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate_tool_chain)
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate_tool_chain")
    graph_builder.add_edge("generate_tool_chain", END)

    graph = graph_builder.compile()

    print("Welcome to the chat! Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        for step in graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                stream_mode="values",
        ):
            step["messages"][-1].pretty_print()


if __name__ == '__main__':
    main()
