# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from app.retriever import retrieve
from app.vector_store import init_vector_store

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    store_name = "chroma_db_doc"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    persistent_directory = os.path.join(base_dir, "vector_db", store_name)
    vector_store = init_vector_store(document_path="knowledge-base-doc", db_name=store_name)

    llm = ChatOllama(temperature=0.1, model="gemma3:latest")


    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs


    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}


    # Step 2: Execute the retrieval.
    tools = ToolNode([retrieve])


    # Step 3: Generate a response using the retrieved content.
    def generate(state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
               or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = llm.invoke(prompt)
        return {"messages": [response]}


    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile()

    input_message = "Hello"

    for step in graph.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
