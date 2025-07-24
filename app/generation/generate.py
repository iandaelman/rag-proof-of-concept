from dotenv import load_dotenv
from langgraph.graph import MessagesState

from app.utils.configuration import get_response_model
from resources.test_data import ragas_data_set

load_dotenv()

GENERATE_PROMPT = (
    "You are a helpful assistant supporting users with their MyMinfin IT-related questions.\n"
    "Based on the following context, please provide a clear and complete answer.\n"
    "If the answer is not available in the context, kindly let the user know that you don't have enough information.\n"
    "Always respond in the same language this question {question} is asked, even if the context is in a different language.\n\n"
    "Question: {question}\n"
    "Context: {context}"
)

response_model = get_response_model()


def generate_answer(state: MessagesState) -> MessagesState:
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return MessagesState(messages=[response])


def generate_answer_with_evaluation(state: MessagesState) -> MessagesState:
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    evaluate_answer(question, context, response.content)
    return MessagesState(messages=[response])


def evaluate_answer(question: str, context: str, answer: str):
    question_index = ragas_data_set["user_input"].index(question)
    for key in ["retrieved_contexts", "answer"]:
        while len(ragas_data_set[key]) < len(ragas_data_set["user_input"]):
            ragas_data_set[key].append("")

    # Set context and answer at the correct index
    ragas_data_set["retrieved_contexts"][question_index] = context
    ragas_data_set["answer"][question_index] = answer if answer is not None else ""
