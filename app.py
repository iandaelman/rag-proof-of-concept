import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
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
    workflow.add_conditional_edges("generate_query_or_respond", tools_condition)
    workflow.add_conditional_edges("tools", grade_documents, {
        "generate_answer": "generate_answer",
        "rewrite_question": "rewrite_question"
    })
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()

graph = build_graph()

st.set_page_config(page_title="Myminfin Support bot", page_icon="🤖")
st.title("Myminfin Support bot")

# Initieer chatgeschiedenis
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Toon eerdere berichten
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input via chat
prompt = st.chat_input("Typ je vraag...")

if prompt:
    # Voeg uservraag toe
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generatin answer..."):
            input_message = HumanMessage(content=prompt)
            message_list = [input_message]

            try:
                output = ""
                for chunk in graph.stream({"messages": message_list}, stream_mode="updates"):
                    for update in chunk.values():
                        for msg in update["messages"]:
                            if isinstance(msg, AIMessage):
                                output += msg.content + "\n"
                st.markdown(output.strip())
                st.session_state.chat_history.append({"role": "assistant", "content": output.strip()})
            except Exception as e:
                st.error(f"Er ging iets mis: {e}")
