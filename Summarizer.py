from huggingface_hub import InferenceClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI
from bs4 import BeautifulSoup
import requests
import uvicorn # Needed for local run
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# 1. ADD CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Allow requests from your local React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------

# Data model for incoming request
class SummaryRequest(BaseModel): 
    url: str

Summary = {} 

# NOTE: Environment variables HF_TOKEN and OPENAI_API_KEY must be set locally.
HF_TOKEN = "HF_TOKEN"

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

try:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.")
        raise KeyError("HF_TOKEN")
    if not OPENAI_KEY:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        raise KeyError("OPENAI_API_KEY")

    # Initialize clients, explicitly using the environment variable values
    HF_client = InferenceClient(token=HF_TOKEN, timeout=120.0) 
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_KEY) 
    print("SUCCESS: API clients initialized for Hugging Face and OpenAI.")

except KeyError as e:
    # If this fails, the global client variables might not be set, relying on runtime checks.
    print(f"CLIENT INIT FAILURE: Missing environment variable {e}. Check your shell session.")
except Exception as e:
    print(f"ERROR initializing API clients: {e}.")
    # Clients may be partially initialized, but we will rely on runtime checks.

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

@app.post("/summarization")
def post_data(request_data: SummaryRequest):
    # Check if necessary keys are present
    if not os.environ.get("HF_TOKEN") or not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500, 
            detail="Server keys missing. Ensure HF_TOKEN and OPENAI_API_KEY are set in your local environment."
        )

    target_url = request_data.url
    
    # 1. Web Scraping
    try:
        article_response = requests.get(target_url, headers=headers)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Web request failed: {e}")

    if article_response.status_code not in (200, 304):
        if article_response.status_code in (401, 403):
            raise HTTPException(status_code=401, detail="Access denied to the article URL (401/403).")
        raise HTTPException(status_code=article_response.status_code, detail=f"Failed to fetch article. Status code: {article_response.status_code}")

    # 2. Extract Text
    soup = BeautifulSoup(article_response.text, "html.parser")
    paragraphs = soup.find_all('p')
    article_text = " ".join([p.get_text() for p in paragraphs])
    
    if not article_text.strip() or len(article_text) < 50:
        raise HTTPException(status_code=400, detail="Error extracting sufficient article text from the URL.")
    
    # 3. HF Summarization (Using InferenceClient)
    try:
        global HF_client
        summarization_result = HF_client.summarization(
            article_text, 
            model="facebook/bart-large-cnn"
        )
        hf_summary = summarization_result[0]['summary_text'] 
    except Exception as e:
        print(f"Hugging Face API Error: {e}")
        # This will now print the full error details for Hugging Face.
        raise HTTPException(status_code=401, detail=f"Hugging Face API Authentication Error (Check HF_TOKEN): {e}")
    
    # 4. OpenAI Summarization
    try:
        global client
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise summarization assistant. Provide a brief summary."},
                {"role": "user", "content": f"summarize the following article: {article_text}"}
            ],
            max_tokens=50,
        )
        openai_summary = response.choices[0].message.content
    except Exception as e:
        # --- CRITICAL CHANGE HERE ---
        # The error is likely here. We print a very specific message and raise a 403 (Forbidden)
        print(f"--- FAILED AT OPENAI CALL ---")
        print(f"OpenAI API Exception Details: {e}")
        print(f"-----------------------------")
        raise HTTPException(status_code=403, detail=f"OpenAI API Forbidden Error. Check OPENAI_API_KEY validity and usage limits: {e}")

    # 5. Store and Return Results
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
        raise HTTPException(status_code=404, detail=f"URL '{url_key}' not found.")

# --- Local Development Entry Point ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)
          






