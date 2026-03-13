from fastapi import FastAPI
import openai
import os, requests
from dotenv import load_dotenv
from pageindex import PageIndexClient
import pageindex.utils as utils

load_dotenv()  # loads .env
app = FastAPI()
pi_client = PageIndexClient(api_key=os.getenv("PAGEINDEX_API_KEY"))
PDF_PATH = "./js.pdf"



@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/chat/{question}")
async def chat(question: str):
    answer = await call_llm(question)
    return {"question": question, "answer": answer}

@app.get("/index-pdf")
def index_pdf():
    if not os.path.exists(PDF_PATH):
        return {"error": "js.pdf not found"}
    # Submit document to PageIndex
    result = pi_client.submit_document(PDF_PATH)
    doc_id = result["doc_id"]
    return {
        "message": "Document submitted for indexing",
        "doc_id": doc_id
    }
@app.get("/tree/{doc_id}")
def get_tree(doc_id: str):
    if not pi_client.is_retrieval_ready(doc_id):
        return {"status": "processing"}
    tree = pi_client.get_tree(doc_id, node_summary=True)["result"]
    return {
        "status": "ready",
        "tree": tree
    }

async def call_llm(prompt, model="gpt-4.1", temperature=0):
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()