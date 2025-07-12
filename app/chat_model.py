from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()


def get_evaluation_model():
    return ChatOllama(model="gemma3n:e4b", temperature=0)

def get_response_model():
    return ChatOllama(model="llama3.2", temperature=0)
    #return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_embedding():
    return OllamaEmbeddings(model='mxbai-embed-large')
    #return OpenAIEmbeddings(model="text-embedding-3-small")
