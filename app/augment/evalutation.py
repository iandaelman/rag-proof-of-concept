from datasets import Dataset
from ragas import RunConfig, evaluate
from ragas.metrics import Faithfulness, FactualCorrectness, AnswerCorrectness, AnswerAccuracy

from app.utils.configuration import get_evaluation_model, response_model_name
from resources.test_data import ragas_data_set


def evaluate_answers():
    normalize_contexts(ragas_data_set)
    dataset = Dataset.from_dict(ragas_data_set)
    my_run_config = RunConfig(max_workers=64, timeout=6000)
    score = evaluate(dataset, metrics=[Faithfulness(),
                                       FactualCorrectness(),
                                       AnswerCorrectness(),
                                       AnswerAccuracy()],
                     llm=get_evaluation_model(), run_config=my_run_config)

    df = score.to_pandas()
    clean_response_model_name = response_model_name.split(":")[0]
    file_name = f"test_results/{clean_response_model_name}_test_results.csv"
    df.to_csv(file_name, index=False)


def normalize_contexts(ragas_data_set):
    contexts = ragas_data_set.get("retrieved_contexts", [])
    normalized = []
    for item in contexts:
        if isinstance(item, list):
            normalized.append(item)
        elif isinstance(item, str):
            # wrap single string as list if not empty
            normalized.append([item] if item else [])
        else:
            normalized.append([])
    ragas_data_set["retrieved_contexts"] = normalized
