from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

# response_model = ChatOllama(model="granite3.3:8b", temperature=0)
response_model = init_chat_model(model="openai:gpt-4o-mini", temperature=0)


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}