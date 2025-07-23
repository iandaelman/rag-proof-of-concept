from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from app.utils.RetrievalMethod import RetrievalMethod

load_dotenv()

#Werkt niet (maakt geen toolcall)
#response_model_name = "granite3.3:8b"
#response_model_name = "llama3.1:8b"
#response_model_name = "llama3.1:8b-instruct-q4_K_M"
#response_model_name = "mistral:7b"


#Is functioneel
#response_model_name = "qwen2.5:7b-instruct"

#Werkt zoals verwacht


#response_model_name = "qwen3:8b"
response_model_name = "llama3.2"


def get_evaluation_model():
    return ChatOpenAI(model="gpt-4o-mini")


def get_response_model():
    return ChatOllama(model=response_model_name, temperature=0)
    # return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_embedding():
    return OllamaEmbeddings(model='mxbai-embed-large')
    #return OpenAIEmbeddings(model="text-embedding-3-small")


def get_retrieve_config(retrieval_method: RetrievalMethod):
    if retrieval_method == RetrievalMethod.MMR:
        return "mmr", {"k": 3, "fetch_k": 20, "lambda_mult": 0.5}
    elif retrieval_method == RetrievalMethod.SIMILARITY_SEARCH:
        return "similarity", {"k": 4}
    elif retrieval_method == RetrievalMethod.SIMILARITY_SCORE_THRESHOLD:
        return "similarity_score_threshold", {"score_threshold": 0.1}

    return "similarity", {"k": 4}
