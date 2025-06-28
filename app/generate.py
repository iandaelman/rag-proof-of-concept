from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState

from app.utils.State import State


def generate(state: State):
    print("\n--- Generating awnser ---")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt_template = ChatPromptTemplate([
        ("system", "You are an assistant for question-answering tasks."),
        ("human", """Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Question: {query} 
        Context: {context}
        Please provide an answer based only on the provided context. If the answer is not found in the context, respond with 'I'm not sure'.""")])
    print("\n--- Initializing LLM model ---")
    llm = ChatOllama(temperature=0.1, model=state["model_name"])
    messages = prompt_template.invoke({"query": state["query"], "context": docs_content})
    print(f"\n--- Invoking LLM model with {messages} ---")
    response = llm.invoke(messages)
    return {"response": response}


# Step 3: Generate a response using the retrieved content.
def generate_tool_chain(state: MessagesState):
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

    llm = ChatOllama(temperature=0.1, model="llama3.2")
    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}
