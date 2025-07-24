from typing import Literal

from dotenv import load_dotenv
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from app.utils.configuration import get_response_model
from app.utils.prompts import GRADE_PROMPT

load_dotenv()




class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


grader_model = get_response_model()


def grade_documents(
        state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""

    # Shortcut: If too many messages (multiple rewrites), stop rewriting
    if len(state["messages"]) >= 5:
        return "generate_answer"

    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


def grade_documents_with_evaluation(
        state: MessagesState,
) -> Literal["generate_answer_with_evaluation", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""

    # Shortcut: If too many messages (e.g. multiple rewrites), stop rewriting
    if len(state["messages"]) >= 5:
        return "generate_answer_with_evaluation"

    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer_with_evaluation"
    else:
        return "rewrite_question"
