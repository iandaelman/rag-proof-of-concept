from langchain_core.messages import RemoveMessage
from langgraph.graph import MessagesState

from app.utils.configuration import get_response_model
from app.utils.prompts import REWRITE_PROMPT

response_model = get_response_model()


def rewrite_question(state: MessagesState) -> MessagesState:
    """
    Rewrite the original user question, based on the cleaned-up conversation context.
    """
    messages = state["messages"]

    messages_to_remove = messages[-2:] #contains retrieved documents and tool call message
    messages_to_keep = messages[:-2]
    original_question = messages[0].content

    messages_to_delete = [RemoveMessage(id=m.id) for m in messages_to_remove]
    questions = "\n".join(m.content for m in messages_to_keep)

    prompt = REWRITE_PROMPT.format(questions=questions, original_question=original_question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return MessagesState(messages=[*messages_to_delete, response])
