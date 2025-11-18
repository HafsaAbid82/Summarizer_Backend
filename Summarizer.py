from huggingface_hub import InferenceClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI
from bs4 import BeautifulSoup
import requests
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://summarizer-eight-pearl.vercel.app",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class SummaryRequest(BaseModel): 
    url: str
Summary = {} 
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
HF_client = None
client = None
try:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set. API will fail.")
    else:
        HF_client = InferenceClient(token=HF_TOKEN, timeout=120.0) 
    if not OPENAI_KEY:
        print("ERROR: OPENAI_API_KEY environment variable is not set. API will fail.")
    else:
        client = OpenAI(api_key=OPENAI_KEY) 
    print("SUCCESS: Attempted API client initialization.")
except Exception as e:
    print(f"ERROR initializing API clients: {e}.")    
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"message": "FastAPI Summarization Service is running on Vercel."}
@app.post("/summarization")
def post_data(request_data: SummaryRequest):
    if not HF_client or not client:
        raise HTTPException(
            status_code=500, 
            detail="Server API keys missing. Ensure HF_TOKEN and OPENAI_API_KEY are set as secrets in Vercel."
        )
    target_url = request_data.url
    try:
        article_response = requests.get(target_url, headers=headers, timeout=15)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Web request failed: {e}")
    if article_response.status_code != 200:
        if article_response.status_code in (401, 403):
            raise HTTPException(status_code=401, detail="Access denied to the article URL (401/403).")
        raise HTTPException(status_code=article_response.status_code, detail=f"Failed to fetch article. Status code: {article_response.status_code}")
    soup = BeautifulSoup(article_response.text, "html.parser")
    article_text = soup.find('body').get_text(separator=' ', strip=True)
    if not article_text.strip() or len(article_text) < 50:
        raise HTTPException(status_code=400, detail="Error extracting sufficient article text from the URL.")
    hf_summary = "HF Summary Failed"
    try:
        summarization_result = HF_client.summarization(
            article_text, 
            model="facebook/bart-large-cnn"
        )
        hf_summary = summarization_result[0]['summary_text'] 
    except Exception as e:
        print(f"Hugging Face API Error: {e}")
        pass
    openai_summary = "OpenAI Summary Failed"
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise summarization assistant. Provide a brief summary, no more than three sentences long."},
                {"role": "user", "content": f"summarize the following article: {article_text}"}
            ],
            max_tokens=50,
        )
        openai_summary = response.choices[0].message.content
    except Exception as e:
        print(f"--- FAILED AT OPENAI CALL ---")
        print(f"OpenAI API Exception Details: {e}")
        print(f"-----------------------------")
        pass
    if hf_summary == "HF Summary Failed" and openai_summary == "OpenAI Summary Failed":
        raise HTTPException(status_code=500, detail="Both Hugging Face and OpenAI summarization attempts failed.")
    new_result = {
        "source_url": target_url,
        "hf_summary": hf_summary,
        "openai_summary": openai_summary,
    }
    Summary[target_url] = new_result
    return {
        "status": "Success",
        "source_url": target_url,
        "hf_summary": hf_summary,
        "openai_summary": openai_summary,
    }
def check_for_existing_result(url_key: str):
    return Summary.get(url_key)
@app.get("/summarization/{url_key:path}")
def get_summary(url_key: str):
    result = check_for_existing_result(url_key)
    if result:
        return {
            "status": "Success",
            "source_url": result["source_url"],
            "hf_summary": result["hf_summary"],
            "openai_summary": result["openai_summary"],
        }
    else:
        raise HTTPException(status_code=404, detail=f"URL '{url_key}' not found in current session memory.")
          






