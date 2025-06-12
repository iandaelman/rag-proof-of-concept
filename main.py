from langgraph.constants import START
from langgraph.graph import StateGraph

from app.gradio_ui import init_chat_chain, chat
from app.retriever import retrieve, generate
from app.state import State
import gradio as gr

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    init_chat_chain(graph)  # this sets the global graph before Gradio launches
    view = gr.ChatInterface(chat, type="messages")
    view.launch(inbrowser=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
