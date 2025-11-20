from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional
import os
import json
import asyncio
from contextlib import asynccontextmanager

# Third-party imports
from huggingface_hub import AsyncInferenceClient
from openai import AsyncOpenAI
from upstash_redis import Redis
from upstash_ratelimit import Ratelimit
from bs4 import BeautifulSoup
import httpx

# --- Configuration ---
REDIS_URL = os.environ.get("Redis_URL")
REDIS_TOKEN = os.environ.get("Redis_Token")
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

# --- Global Clients ---
# We use a lifespan manager to handle resources efficiently
redis_client: Optional[Redis] = None
ratelimit: Optional[Ratelimit] = None
hf_client: Optional[AsyncInferenceClient] = None
openai_client: Optional[AsyncOpenAI] = None
http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize clients on startup and clean them up on shutdown.
    """
    global redis_client, ratelimit, hf_client, openai_client, http_client
    
    # 1. Initialize Redis
    if REDIS_URL and REDIS_TOKEN:
        try:
            # Upstash Python SDK is HTTP-based and sync by default, 
            # but fast enough for this use case. 
            redis_client = Redis(url=REDIS_URL, token=REDIS_TOKEN)
            ratelimit = Ratelimit(
                redis=redis_client,
                limiter=Ratelimit.sliding_window(4, "1 m"),
            )
            print("SUCCESS: Redis and Ratelimit initialized.")
        except Exception as e:
            print(f"WARNING: Redis init failed: {e}")
    
    # 2. Initialize AI Clients
    if HF_TOKEN:
        hf_client = AsyncInferenceClient(token=HF_TOKEN)
    
    if OPENAI_KEY:
        openai_client = AsyncOpenAI(api_key=OPENAI_KEY)

    # 3. Shared HTTP Client for Scraping (Prevent opening new connection per request)
    http_client = httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0 (SummarizerBot/1.0)"},
        timeout=15.0,
        follow_redirects=True
    )

    yield # App runs here

    # Cleanup
    await http_client.aclose()
    print("Shutdown: Resources released.")

app = FastAPI(lifespan=lifespan)

# --- Middleware ---
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

# --- Models ---
class SummaryRequest(BaseModel):
    url: HttpUrl # Validates that input is actually a URL

# --- Helper Functions ---
async def fetch_and_clean_article(url: str) -> str:
    """
    Async fetching and smarter cleaning of HTML.
    """
    try:
        response = await http_client.get(url)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (401, 403):
            raise HTTPException(status_code=403, detail="Access denied to article (401/403).")
        raise HTTPException(status_code=e.response.status_code, detail="Failed to fetch article.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unwanted elements (nav, footer, scripts, styles)
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        element.decompose()

    # Extract text primarily from Paragraphs to avoid menu items
    paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 20]
    article_text = " ".join(paragraphs)

    if not article_text or len(article_text) < 50:
        # Fallback to body if p tags fail
        article_text = soup.body.get_text(separator=' ', strip=True)

    if len(article_text) < 50:
        raise HTTPException(status_code=400, detail="Could not extract sufficient text content.")

    # Limit to ~600 words for inputs
    return " ".join(article_text.split()[:600])

# --- Routes ---
@app.get("/")
def read_root():
    return {"message": "FastAPI Summarization Service is running."}

@app.post("/summarization")
async def post_data(request_data: SummaryRequest, request: Request):
    url_str = str(request_data.url)

    # 1. Rate Limiting
    if ratelimit:
        user_ip = request.headers.get("x-forwarded-for", request.client.host)
        # Note: Upstash rate limit is sync, but lightweight
        limit = ratelimit.limit(user_ip)
        if not limit["allowed"]:
            raise HTTPException(
                status_code=429, 
                detail=f"Rate limit exceeded. Reset in {limit['reset']}s"
            )

    # 2. Cache Check
    if redis_client:
        try:
            cached = redis_client.get(url_str)
            if cached:
                data = json.loads(cached)
                return {
                    "status": "Success (Cached)",
                    "source_url": url_str,
                    "hf_summary": data.get("hf_summary"),
                    "openai_summary": data.get("openai_summary")
                }
        except Exception:
            pass # Fail silently on cache read error

    # 3. Scrape Article (Async)
    truncated_text = await fetch_and_clean_article(url_str)

    # 4. Perform Summarization (Parallel Execution optional, currently sequential for safety)
    
    # -- Hugging Face --
    hf_summary = "HF Summary Failed"
    hf_error = None
    if hf_client:
        try:
            # Using facebook/bart-large-cnn
            result = await hf_client.summarization(
                truncated_text, 
                model="facebook/bart-large-cnn",
                parameters={"truncation": "only_first"}
            )
            # AsyncInferenceClient might return a list or object depending on version
            hf_summary = result.summary_text if hasattr(result, 'summary_text') else result[0]['summary_text']
        except Exception as e:
            hf_error = str(e)
            print(f"HF Error: {e}")

    # -- OpenAI --
    openai_summary = "OpenAI Summary Failed"
    openai_error = None
    if openai_client:
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a concise summarization assistant. Keep it under 3 sentences."},
                    {"role": "user", "content": f"Summarize this: {truncated_text}"}
                ],
                max_tokens=100,
                temperature=0.1,
            )
            openai_summary = response.choices[0].message.content
        except Exception as e:
            openai_error = str(e)
            print(f"OpenAI Error: {e}")

    # 5. Error Handling
    if hf_summary == "HF Summary Failed" and openai_summary == "OpenAI Summary Failed":
        detail_msg = "Both providers failed."
        if hf_error: detail_msg += f" HF: {hf_error}."
        if openai_error: detail_msg += f" OpenAI: {openai_error}."
        raise HTTPException(status_code=500, detail=detail_msg)

    # 6. Cache Result
    if redis_client:
        try:
            redis_client.set(
                url_str,
                json.dumps({
                    "hf_summary": hf_summary,
                    "openai_summary": openai_summary
                }),
                ex=3600
            )
        except Exception as e:
            print(f"Cache write failed: {e}")

    return {
        "status": "Success",
        "source_url": url_str,
        "hf_summary": hf_summary,
        "openai_summary": openai_summary
    }

@app.get("/summarization/{url_key:path}")
def get_summary(url_key: str):
    """
    Retrieve a specific URL summary from cache explicitly.
    """
    if not redis_client:
         raise HTTPException(status_code=503, detail="Cache service unavailable.")
         
    cached = redis_client.get(url_key)
    if cached:
        data = json.loads(cached)
        return {
            "status": "Success",
            "source_url": url_key,
            "hf_summary": data.get("hf_summary"),
            "openai_summary": data.get("openai_summary")
        }
    
    raise HTTPException(status_code=404, detail="URL not found in cache.")
