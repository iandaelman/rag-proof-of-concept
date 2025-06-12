# This is a sample Python script.
from app.document_loader import load_documents_from_knowledge_base, split_documents
from app.embedder import create_datastore
from app.gradio_ui import init_chat_chain, chat
from app.retriever import initialize_llm

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    documents = load_documents_from_knowledge_base()
    chunks = split_documents(documents=documents)
    vectorstore = create_datastore(chunks)
    llm, memory = initialize_llm("llama3.2")
    retriever = vectorstore.as_retriever()

    init_chat_chain(llm, retriever, memory)

    question = "Qu'est-ce qu'un switchMap ?"
    answer = chat(question)
    print(answer)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
