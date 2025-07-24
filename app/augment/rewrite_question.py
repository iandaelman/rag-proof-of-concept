from langchain_core.messages import RemoveMessage
from langgraph.graph import MessagesState

from app.utils.configuration import get_response_model

REWRITE_PROMPT = (
    "You are assisting with improving a user's question related to Myminfin IT support.\n"
    "Conversation history so far (most recent last):"
    "\n ------- \n"
    "{questions}"
    "\n ------- \n"
    "Original question:"
    "\n ------- \n"
    "{original_question}"
    "\n ------- \n"
    "Rewrite the last question to make it clearer and more specific, without changing its original meaning or intent.\n"
    "- If the question is in Dutch, rewrite and translate it to French.\n"
    "- If the question is in French, rewrite and translate it to English.\n"
    "- Do not repeat or restate earlier questions exactly.\n"
    "- If the last question is very similar to a previous one, improve it by adding useful clarifications or context relevant to Myminfin IT support.\n"
    "- Return only the rewritten question, nothing else."
)

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
