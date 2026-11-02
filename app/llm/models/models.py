import os
import getpass


from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_cohere import CohereEmbeddings, CohereRerank,ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

model = ChatOllama(
    base_url="https://ollama.com",
    model="gemma4:cloud",
    temperature=0,
    client_kwargs={
        "headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"},
    },
)


model3 = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=os.getenv('GOOGLE_API_KEY')) 


if not os.environ.get("COHERE_API_KEY"):
    os.environ["COHERE_API_KEY"] = getpass.getpass("Enter API key for Cohere: ")

model2 = ChatCohere(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model=os.getenv("COHERE_EMBED_MODEL", "command-r7b-12-2024"),
) 

embeddings = CohereEmbeddings(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model=os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0"),
)

reranker = CohereRerank(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model=os.getenv("COHERE_RERANKER_MODEL", "rerank-english-v2.0"),
    top_n=10,
)



# Requires JINA_API_KEY in .env; without it Jina rejects the request with 401.
jina_api_key = os.getenv("JINA_API_KEY")

jina_reranker ={"url" :"https://api.jina.ai/v1/rerank" , "headers" : {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {jina_api_key}" if jina_api_key else ""
} , "data" : {"model" :"jina-reranker-v3.5" , "top_n": 3,}}

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


if __name__ == "__main__":
    result = model3.invoke("What's the weather in San Francisco?")
    print(result.content)

    get_weather(city="San Francisco")