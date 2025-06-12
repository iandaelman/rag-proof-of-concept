# This is a sample Python script.
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from app.document_loader import load_documents_from_knowledge_base, split_documents
from app.embedder import create_datastore
from app.retriever import initialize_llm

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    documents = load_documents_from_knowledge_base()
    chunks = split_documents(documents=documents)
    vectorstore = create_datastore(chunks)
    llm, memory = initialize_llm()
    retriever = vectorstore.as_retriever()
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    query = "Please explain what Insurellm is in a couple of sentences"
    result = conversation_chain.invoke({"question": query})
    print(result["answer"])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
