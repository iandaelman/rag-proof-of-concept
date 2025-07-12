from datasets import Dataset
from dotenv import load_dotenv
from langgraph.graph import MessagesState
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness, LLMContextRecall, Faithfulness, \
    FactualCorrectness

from app.chat_model import get_response_model, get_evaluation_model
from app.test_data import test_data

load_dotenv()

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Always answer in the language this question: {question} is asked no matter the language of the context provided. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

response_model = get_response_model()


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    evaluate_answer(question, context, response.content)
    return {"messages": [response]}


def evaluate_answer(question: str, context: str, answer: str):
    question_index = test_data["user_input"].index(question)
    for key in ["retrieved_contexts", "answer"]:
        while len(test_data[key]) < len(test_data["user_input"]):
            test_data[key].append("")


    # Set context and answer at the correct index
    test_data["retrieved_contexts"][question_index] = [context]
    test_data["answer"][question_index] = answer

    dataset = Dataset.from_dict(test_data)
    my_run_config = RunConfig(max_workers=64, timeout=6000)
    score = evaluate(dataset, metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
                     llm=get_evaluation_model(), run_config=my_run_config)

    df = score.to_pandas()
    df.to_csv('test.csv', index=False)

