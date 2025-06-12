from langgraph.graph.state import CompiledStateGraph

graph: CompiledStateGraph | None = None

def init_chat_chain(g: CompiledStateGraph):
    global graph
    graph = g


def chat(question, history=None):
    if graph is None:
        raise ValueError("Chat chain is not initialized. Call init_chat_chain() first.")
    result = graph.invoke({"question": question})
    return result["answer"]
