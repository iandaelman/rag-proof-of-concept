from dotenv import load_dotenv
from langgraph.graph import MessagesState

from app.utils.configuration import get_response_model
from resources.test_data import ragas_data_set

load_dotenv()

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Always answer in the language this question: {question} is asked no matter the language of the context provided. "
    "Give all relevant information"
    "Question: {question} \n"
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
