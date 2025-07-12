from datasets import Dataset
from ragas import RunConfig, evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

from app.chat_model import get_evaluation_model, response_model_name
from app.test_data import ragas_data_set


def evaluate_answers():
    dataset = Dataset.from_dict(ragas_data_set)
    my_run_config = RunConfig(max_workers=64, timeout=6000)
    score = evaluate(dataset, metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
                     llm=get_evaluation_model(), run_config=my_run_config)

    df = score.to_pandas()
    file_name = f"test_results/{response_model_name}_test_results.csv"
    df.to_csv(file_name, index=False)
