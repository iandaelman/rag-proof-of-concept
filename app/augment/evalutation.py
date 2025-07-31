import os

from datasets import Dataset
from ragas import RunConfig, evaluate
from ragas.metrics import Faithfulness, AnswerCorrectness, LLMContextRecall, ResponseRelevancy

from app.utils.configuration import get_evaluation_model, response_model_name
from resources.test_data import ragas_data_set


def evaluate_answers():
    normalize_contexts(ragas_data_set)
    dataset = Dataset.from_dict(ragas_data_set)
    my_run_config = RunConfig(max_workers=64, timeout=6000)
    score = evaluate(dataset, metrics=[Faithfulness(), # measures how factually consistent a response is with the retrieved context
                                       ResponseRelevancy(), #  how relevant a response is to the user input, An answer is considered relevant if it directly and appropriately addresses the original question.
                                       AnswerCorrectness(), #Measures answer correctness compared to ground truth as a combination of factuality and semantic similarity.
                                       LLMContextRecall()], #Measures how well the retrieved context includes all the necessary information to answer the question by estimating the proportion of relevant facts found (true positives) versus missing facts (false negatives).
                     llm=get_evaluation_model(), run_config=my_run_config)

    df = score.to_pandas()
    clean_response_model_name = response_model_name.split(":")[0]
    file_name = f"test_results/{clean_response_model_name}_test_results.csv"
    os.makedirs("test_results", exist_ok=True)
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
