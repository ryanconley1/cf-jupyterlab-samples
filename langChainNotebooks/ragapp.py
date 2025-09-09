from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_postgres.vectorstores import PGVector
import requests
import os
import requests
import json
import httpx
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain_core.tools import Tool
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from datetime import date
from langchain_nomic import NomicEmbeddings
import warnings
import ssl
from langchain_community.embeddings import OllamaEmbeddings
from openai import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
import re

# -----------------------------
# Embedding setup
# -----------------------------
# -----------------------------
# Load services from env
# -----------------------------
vcapservices = os.getenv('VCAP_SERVICES')
services = json.loads(vcapservices)

# -----------------------------
# Embedding service details
# -----------------------------
def is_embeddingservice(service):
    return service["name"] == "prod-embedding-nomic-text"

embedding_services = filter(is_embeddingservice, services["genai"])
embedding_credentials = list(embedding_services)[0]["credentials"]

API_BASE = embedding_credentials["api_base"] + "/v1"
API_KEY = embedding_credentials["api_key"]
MODEL_NAME = embedding_credentials["model_name"]

def embed_text(text: str):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,   # make sure MODEL_NAME is a str, not a tuple
        "input": text
    }

    print("HEADERS:", headers)
    print("PAYLOAD:", payload)

    resp = requests.post(
        f"{API_BASE}/embeddings",
        headers=headers,
        json=payload,
        verify=False
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]

class CustomEmbeddings:
    def embed_documents(self, texts): return [embed_text(t) for t in texts]
    def embed_query(self, text): return embed_text(text)

embedding = CustomEmbeddings()

# -----------------------------
# Vectorstore setup
# -----------------------------
def is_vectordbservice(service):
    return service["name"] == "vector-db"
db_services = filter(is_vectordbservice, services["postgres"])
db_credentials = list(db_services)[0]["credentials"]
DB_URI = db_credentials["uri"]

vectorstore = PGVector(
    embeddings=embedding,
    connection=DB_URI,
    collection_name="maintenance_and_taxonomy",
    use_jsonb=True,
    create_extension=False,   # already created
)

# -----------------------------
# RAG setup
# -----------------------------
httpx_client = httpx.Client(http2=True, verify=False, timeout=30.0)

vcapservices = os.getenv('VCAP_SERVICES')
services = json.loads(vcapservices)

def is_chatservice(service):
    return service["name"] == "gen-ai-qwen3-ultra"

chat_services = filter(is_chatservice, services["genai"])
chat_credentials = list(chat_services)[0]["credentials"]


llm = ChatOpenAI(temperature=0.9, model=chat_credentials["model_name"], base_url=chat_credentials["api_base"], api_key=chat_credentials["api_key"], http_client=httpx_client)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


# -----------------------------
# Gradio chatbot function
# -----------------------------
def sanitize_answer(answer):
    # Convert sets to lists
    if isinstance(answer, set):
        answer = list(answer)
    # Convert anything else not string/dict/list to string
    if not isinstance(answer, (str, dict, list)):
        answer = str(answer)
    # Remove <think> tags if present
    if isinstance(answer, str):
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    return answer

def predict(message, history):
    # Run RetrievalQA
    output = qa.invoke({"query": message})  # dict: {"result", "source_documents"}
    answer = sanitize_answer(output["result"])
    return answer
    
demo = gr.ChatInterface(
    fn=predict,
    title="🛠 Aircraft Maintenance Chatbot",
    description="Ask questions about maintenance records or aircraft taxonomy."
)


demo.launch(
    share=True,
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True,
    quiet=False
)