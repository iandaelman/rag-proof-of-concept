from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama


def initialize_llm(model="llama3.2"):
    llm = ChatOllama(model=model, temperature=7)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    return llm, memory
