from typing import Literal

from dotenv import load_dotenv
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from app.chat_model import get_response_model

load_dotenv()

response_model = get_response_model()

QUERY_OR_RESPOND_PROMPT = """
This method decides whether to call the retriever tool or respond directly.

If the user's question is trivial, respond directly.  
If the question is non-trivial, use the retriever tool to generate a response.

Given the user's question:  
"{question}"

Determine whether the question is trivial.  
Return a binary score: 'yes' if it is trivial, 'no' if it is not.
"""


class GenerateQueryOrRespond(BaseModel):
    binary_score: str = Field(
        description="Query is a trivial question: 'yes' if it's a trivial question, or 'no' if not"
    )


def generate_query_or_respond(state: MessagesState) -> Literal["END", "retrieve"]:
    question = state["messages"][0].content

    prompt = QUERY_OR_RESPOND_PROMPT.format(question=question)

    response = response_model.with_structured_output(GenerateQueryOrRespond).invoke(
        [{"role": "user", "content": prompt}])

    score = response.binary_score
    print("Score: " + score)
    if score == "yes":
        return "END"
    else:
        return "retrieve"
