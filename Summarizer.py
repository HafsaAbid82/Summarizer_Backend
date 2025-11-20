from huggingface_hub import InferenceClient
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI
from bs4 import BeautifulSoup
import requests
import json
from upstash_redis import Redis
from upstash_ratelimit import Ratelimit

# ----------------------------------------------------------------------
# 1. Environment Variable Checks and DEFERRED Initialization (CRITICAL FIX)
#    This ensures the app doesn't crash globally if Redis secrets are missing.
# ----------------------------------------------------------------------

REDIS_URL = os.environ.get("Redis_URL")
REDIS_TOKEN = os.environ.get("Redis_Token")
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

redis = None
ratelimit = None
HF_client = None
client = None

if REDIS_URL and REDIS_TOKEN:
    try:
        # **Caching and Rate Limiting features are initialized HERE**
        redis = Redis(url=REDIS_URL, token=REDIS_TOKEN)
        ratelimit = Ratelimit(
            redis=redis,
            limiter=Ratelimit.sliding_window(2, "1 m"),
        )
        print("SUCCESS: Redis and Ratelimit initialized and active.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize Redis/Ratelimit: {e}")
        # If initialization fails, they remain None.
else:
    print("WARNING: Redis_URL or Redis_Token is missing. Caching and RateLimiting will be DISABLED.")

# ----------------------------------------------------------------------
# 2. API Client Initialization
# ----------------------------------------------------------------------

try:
    if HF_TOKEN:
        # **FIX: Added provider="hf-inference" as requested**
        HF_client = InferenceClient(provider="hf-inference", token=HF_TOKEN, timeout=120.0) 
    else:
        print("ERROR: HF_TOKEN environment variable is not set.")

    if OPENAI_KEY:
        client = OpenAI(api_key=OPENAI_KEY) 
    else:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")

    print("SUCCESS: Attempted API client initialization.")
except Exception as e:
    print(f"ERROR initializing API clients: {e}.")    
    
# ----------------------------------------------------------------------
# 3. FastAPI Setup
# ----------------------------------------------------------------------
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
    
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# ----------------------------------------------------------------------
# 4. Endpoints
# ----------------------------------------------------------------------
@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"message": "FastAPI Summarization Service is running on Vercel."}

@app.post("/summarization")
def post_data(request_data: SummaryRequest, request: Request):
    # Check 1: External API Clients (Must be initialized for summarization to work)
    if not HF_client or not client:
        raise HTTPException(
            status_code=500, 
            detail="Server API clients failed to initialize. Check HF_TOKEN and OPENAI_API_KEY setup."
        )
    
    target_url = request_data.url
    
    # Check 2: Cache Read (FEATURE CHECK: Only runs if Redis was successfully initialized)
    cached = None
    if redis:
        cached = redis.get(target_url)
        if cached:
            if isinstance(cached, bytes):
                cached = cached.decode('utf-8')
            data = json.loads(cached)
            return {
                "status": "Success",
                "source_url": target_url,
                "hf_summary": data.get("hf_summary"),
                "openai_summary": data.get("openai_summary")
            }
        
    # Check 3: Rate Limit (FEATURE CHECK: Only runs if Ratelimit was successfully initialized)
    if ratelimit:
        # Use X-Forwarded-For to get the user's real IP behind Vercel's proxy
        user_ip = request.headers.get("x-forwarded-for", request.client.host)
        limit = ratelimit.limit(user_ip) 
        
        if not limit["allowed"]:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {limit['reset']} seconds."
            )
        
    # --- Web Scraping ---
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
    
    # --- Truncate article text for Hugging Face model ---
    words = article_text.split()
    truncated_text = " ".join(words[:500])
    
    # --- Summarization Logic ---
    hf_summary = "HF Summary Failed"
    hf_error_detail = ""
    try:
        summarization_result = HF_client.summarization(
            truncated_text, 
            model="facebook/bart-large-cnn"
        )
        hf_summary = summarization_result[0]['summary_text'] 
    except Exception as e:
        hf_error_detail = str(e) # Capture the specific error
        print(f"Hugging Face API Error: {e}")
        pass
        
    openai_summary = "OpenAI Summary Failed"
    openai_error_detail = ""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise summarization assistant. Provide a brief summary, no more than three sentences long."},
                {"role": "user", "content": f"summarize the following article: {truncated_text}"}
            ],
            max_tokens=50,
            temperature=0.1,
        )
        openai_summary = response.choices[0].message.content
    except Exception as e:
        openai_error_detail = str(e) # Capture the specific error
        print(f"--- FAILED AT OPENAI CALL ---")
        print(f"OpenAI API Exception Details: {e}")
        pass
        
    if hf_summary == "HF Summary Failed" and openai_summary == "OpenAI Summary Failed":
        # --- RETURN DETAILED ERROR MESSAGE ---
        error_detail = "Both Hugging Face and OpenAI summarization attempts failed."
        if hf_error_detail:
            error_detail += f" HF Error: {hf_error_detail}."
        if openai_error_detail:
            error_detail += f" OpenAI Error: {openai_error_detail}."
            
        raise HTTPException(status_code=500, detail=error_detail)
        
    # Check 4: Cache Write (FEATURE CHECK: Only runs if Redis was successfully initialized)
    if redis:
        redis.set(
            target_url, 
            json.dumps ({
                "hf_summary": hf_summary,
                "openai_summary": openai_summary
            }), 
            ex=3600 
        )
        
    return {
        "status": "Success",
        "source_url": target_url,
        "hf_summary": hf_summary,
        "openai_summary": openai_summary
    }

@app.get("/summarization/{url_key:path}")
def get_summary(url_key: str):
    # Check 5: Cache Read for GET (FEATURE CHECK: Only runs if Redis was successfully initialized)
    if redis:
        cached = redis.get(url_key)
        if cached:
            if isinstance(cached, bytes):
                cached = cached.decode('utf-8')
            data = json.loads(cached)
            return {
                "status": "Success",
                "source_url": url_key,
                "hf_summary": data.get("hf_summary"),
                "openai_summary": data.get("openai_summary")
            }
        
    raise HTTPException(status_code=404, detail=f"URL '{url_key}' not found in session memory.")
