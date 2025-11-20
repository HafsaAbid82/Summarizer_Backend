from huggingface_hub import InferenceClient
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import AsyncOpenAI # Use Async Client
from bs4 import BeautifulSoup
import httpx # Async replacement for requests
import json
import asyncio
from upstash_redis import Redis
from upstash_ratelimit import Ratelimit

# --- Configuration ---
REDIS_URL = os.environ.get("Redis_URL")
REDIS_TOKEN = os.environ.get("Redis_Token")
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

# --- Global Clients ---
redis = None
ratelimit = None
hf_client = None
openai_client = None

# Initialize Redis
if REDIS_URL and REDIS_TOKEN:
    try:
        redis = Redis(url=REDIS_URL, token=REDIS_TOKEN)
        ratelimit = Ratelimit(
            redis=redis,
            limiter=Ratelimit.sliding_window(4, "1 m"),
        )
        print("SUCCESS: Redis and Ratelimit initialized.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize Redis: {e}")
else:
    print("WARNING: Redis credentials missing. Caching disabled.")

# Initialize AI Clients
try:
    if HF_TOKEN:
        # Ideally use AsyncInferenceClient, but standard is fine for now if wrapped
        hf_client = InferenceClient(provider="auto", token=HF_TOKEN, timeout=120.0) 
    if OPENAI_KEY:
        openai_client = AsyncOpenAI(api_key=OPENAI_KEY) # Async Client
    print("SUCCESS: AI clients initialized.")
except Exception as e:
    print(f"ERROR initializing AI clients: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://summarizer-eight-pearl.vercel.app", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummaryRequest(BaseModel): 
    url: str

# --- Helper Functions ---

async def scrape_url(url: str) -> str:
    """Asynchronously fetches and cleans text from a URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                raise HTTPException(status_code=403, detail="Access denied by target website.")
            raise HTTPException(status_code=e.response.status_code, detail="Failed to fetch article.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")

    soup = BeautifulSoup(response.text, "html.parser")
    
    # Remove script and style elements to clean up text
    for script in soup(["script", "style", "header", "footer", "nav", "meta"]):
        script.extract()

    text = soup.get_text(separator=' ', strip=True)
    
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Page content too short to summarize.")
        
    # Truncate to ~500 words to save tokens
    return " ".join(text.split()[:500])

async def get_hf_summary(text: str) -> str:
    """Wrapper to run synchronous HF call in a thread if needed, or use directly."""
    if not hf_client:
        raise Exception("HF Client not initialized")
    
    try:
        # Running sync function in thread pool to prevent blocking event loop
        result = await asyncio.to_thread(
            hf_client.summarization, 
            text, 
            model="facebook/bart-large-cnn"
        )
        return result[0]['summary_text']
    except Exception as e:
        print(f"HF Error: {e}")
        raise e

async def get_openai_summary(text: str) -> str:
    if not openai_client:
        raise Exception("OpenAI Client not initialized")
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise summarization assistant. Max 3 sentences."},
                {"role": "user", "content": f"Summarize this: {text}"}
            ],
            max_tokens=100,
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI Error: {e}")
        raise e

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "FastAPI Summarization Service is running."}

@app.post("/summarization")
async def post_data(request_data: SummaryRequest, request: Request):
    target_url = request_data.url

    # 1. Check Cache
    if redis:
        cached = redis.get(target_url)
        if cached:
            if isinstance(cached, bytes): cached = cached.decode('utf-8')
            return json.loads(cached)

    # 2. Rate Limiting
    if ratelimit:
        # Fallback to host if x-forwarded-for is missing
        user_ip = request.headers.get("x-forwarded-for", request.client.host or "127.0.0.1")
        
        # Upstash limit returns a Response object, access .allowed (boolean)
        limit_response = ratelimit.limit(user_ip) 
        if not limit_response.allowed:
             raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    # 3. Scrape Data
    truncated_text = await scrape_url(target_url)
    
    hf_summary = "Failed"
    openai_summary = "Failed"
    hf_error = ""

    # 4. Attempt Hugging Face
    try:
        hf_summary = await get_hf_summary(truncated_text)
    except Exception as e:
        hf_error = str(e)

    # 5. Attempt OpenAI
    try:
        openai_summary = await get_openai_summary(truncated_text)
    except Exception as e:
        # If both failed, raise error
        if hf_summary == "Failed":
            raise HTTPException(status_code=500, detail=f"All AI services failed. HF Error: {hf_error}. OpenAI Error: {e}")

    response_payload = {
        "status": "Success",
        "source_url": target_url,
        "hf_summary": hf_summary,
        "openai_summary": openai_summary
    }

    # 6. Set Cache
    if redis:
        redis.set(target_url, json.dumps(response_payload), ex=3600)

    return response_payload

@app.get("/summarization/cache")
def get_cache_by_key(url: str):
    # Pass the URL as a query param: /summarization/cache?url=...
    if not redis:
         raise HTTPException(status_code=503, detail="Redis not available")
         
    cached = redis.get(url)
    if cached:
        if isinstance(cached, bytes): cached = cached.decode('utf-8')
        return json.loads(cached)
        
    raise HTTPException(status_code=404, detail="URL not found in cache.")
