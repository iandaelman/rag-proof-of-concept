from langgraph.graph import MessagesState

from app.utils.configuration import get_response_model

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question, if the question is written in dutch rewrite and translate it to french."
    "If the question is written in French rewrite it to English only return the rewritten question nothing else:"
)

response_model = get_response_model()

def rewrite_question(state: MessagesState)-> MessagesState:
    """Rewrite the original user question.
    If the question is in dutch rewrite and translate the question to french.
    If the question is in french rewrite and translate it to English
    """
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return MessagesState(messages=[response])