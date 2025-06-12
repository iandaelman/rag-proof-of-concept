from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.callbacks import StdOutCallbackHandler

conversation_chain: ConversationalRetrievalChain | None = None

def init_chat_chain(llm, retriever, memory):
    global conversation_chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        callbacks=[StdOutCallbackHandler()],
    )

def chat(question, history=None):
    if conversation_chain is None:
        raise ValueError("Chat chain is not initialized. Call init_chat_chain() first.")
    result = conversation_chain.invoke({"question": question})
    return result["answer"]