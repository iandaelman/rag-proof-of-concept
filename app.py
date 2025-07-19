import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from app.augment.GradeDocuments import grade_documents
from app.augment.rewrite_question import rewrite_question
from app.generation.GenerateOrQuery import generate_query_or_respond
from app.generation.generate import generate_answer
from app.retrieve.retriever import retriever_tool

load_dotenv()

@st.cache_resource
def build_graph():
    workflow = StateGraph(MessagesState)
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("tools", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition
    )
    workflow.add_conditional_edges(
        "tools",
        grade_documents,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question"
        }
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()

graph = build_graph()

st.title("📚 Mijn AI Agent Interface")

user_input = st.text_input("💬 Wat is je vraag?")

if st.button("🚀 Verstuur vraag"):
    if user_input.strip():
        st.info("⏳ Verwerken...")
        input_message = HumanMessage(content=user_input)
        message_list = [input_message]

        for chunk in graph.stream({"messages": message_list}, stream_mode="updates"):
            for node, update in chunk.items():
                st.markdown(f"**📍 Node update: `{node}`**")
                for msg in update["messages"]:
                    st.write(msg.content)
    else:
        st.warning("Voer een geldige vraag in.")

