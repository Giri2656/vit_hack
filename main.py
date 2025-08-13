# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import rag_utils
import os
import io
import requests
from datetime import datetime
from typing import List, Optional

load_dotenv()

app = FastAPI(
    title="AI Tutor Backend",
    description="AI Tutor with universal file ingestion + RAG (Gemini) + Resource Recommendation",
    version="1.0.0",
)

# CORS (open for hackathon demo; tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------- Models --------
class AskRequest(BaseModel):
    question: str

class HealthResponse(BaseModel):
    status: str
    message: str
    gemini_configured: bool

class ResourceRecommendationRequest(BaseModel):
    topic: str
    include_youtube: bool = True
    include_pdfs: bool = True
    include_web: bool = True

class SearchRequest(BaseModel):
    query: str
    types: List[str] = ["youtube", "pdfs", "web"]

# -------- Resource Recommendation System --------
YOUTUBE_API_KEY = "AIzaSyAIOx5R8O1oAv97rH0_8W8i5ObI_X0801M"

def fetch_most_viewed_videos(query: str) -> List[dict]:
    """Fetch most viewed YouTube videos for a topic"""
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 5,
            "order": "viewCount",
            "key": YOUTUBE_API_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        videos = []
        for item in data.get("items", []):
            video = {
                "id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"][:100] + "..." if len(item["snippet"]["description"]) > 100 else item["snippet"]["description"],
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                "link": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "channel": item["snippet"]["channelTitle"],
                "published": item["snippet"]["publishedAt"]
            }
            videos.append(video)
        return videos
    except Exception as e:
        print(f"Error fetching most viewed videos: {e}")
        return []

def fetch_top_rated_videos(query: str) -> List[dict]:
    """Fetch top rated YouTube videos for a topic"""
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 5,
            "order": "rating",
            "key": YOUTUBE_API_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        videos = []
        for item in data.get("items", []):
            video = {
                "id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"][:100] + "..." if len(item["snippet"]["description"]) > 100 else item["snippet"]["description"],
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                "link": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "channel": item["snippet"]["channelTitle"],
                "published": item["snippet"]["publishedAt"]
            }
            videos.append(video)
        return videos
    except Exception as e:
        print(f"Error fetching top rated videos: {e}")
        return []

def fetch_pdf_resources(query: str) -> List[dict]:
    """Fetch PDF resources for a topic"""
    try:
        pdf_sources = [
            {
                "title": f"Google Search: {query} PDFs",
                "link": f"https://www.google.com/search?q={query}+filetype:pdf",
                "source": "Google Search",
                "type": "search"
            },
            {
                "title": f"Scholar Search: {query} Academic Papers",
                "link": f"https://scholar.google.com/scholar?q={query}",
                "source": "Google Scholar",
                "type": "academic"
            },
            {
                "title": f"ResearchGate: {query} Research Papers",
                "link": f"https://www.researchgate.net/search/publication?q={query}",
                "source": "ResearchGate",
                "type": "research"
            },
            {
                "title": f"arXiv: {query} Preprints",
                "link": f"https://arxiv.org/search/?query={query}",
                "source": "arXiv",
                "type": "preprint"
            }
        ]
        return pdf_sources
    except Exception as e:
        print(f"Error fetching PDF resources: {e}")
        return []

def fetch_web_resources(query: str) -> List[dict]:
    """Fetch general web resources for a topic"""
    try:
        web_sources = [
            {
                "title": f"Wikipedia: {query}",
                "link": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                "source": "Wikipedia",
                "type": "encyclopedia"
            },
            {
                "title": f"Stack Overflow: {query} Programming Questions",
                "link": f"https://stackoverflow.com/search?q={query}",
                "source": "Stack Overflow",
                "type": "programming"
            },
            {
                "title": f"Reddit: {query} Discussions",
                "link": f"https://www.reddit.com/search/?q={query}",
                "source": "Reddit",
                "type": "community"
            },
            {
                "title": f"Medium: {query} Articles",
                "link": f"https://medium.com/search?q={query}",
                "source": "Medium",
                "type": "articles"
            }
        ]
        return web_sources
    except Exception as e:
        print(f"Error fetching web resources: {e}")
        return []

# -------- Frontend Routes --------
@app.get("/")
async def serve_frontend():
    """Serve the main frontend"""
    return FileResponse("static/index.html")

@app.get("/demo")
async def serve_demo():
    """Serve the demo frontend"""
    return FileResponse("static/index.html")

# -------- Enhanced Routes with Resource Recommendation --------
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="OK",
        message="Service is running with Resource Recommendation",
        gemini_configured=bool(os.getenv("GEMINI_API_KEY")),
    )

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """
    Universal ingestion:
    - Reads upload once (to get size)
    - Resets the stream so rag_utils can read again
    - Extracts text (pdf/doc/ppt/xls/txt/html/css/img via OCR)
    - Chunks + embeds into FAISS for RAG
    """
    try:
        # read once to get size and reset for processing
        raw = await file.read()
        size_bytes = len(raw or b"")
        # reset UploadFile stream so rag_utils can read again
        file.file = io.BytesIO(raw)

        # extract text using universal extraction pipeline
        extracted_text = await rag_utils.universal_text_extraction(file)
        if not extracted_text or not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in the file.")

        # chunk + embed
        chunks = rag_utils.chunk_text(extracted_text, chunk_size=500)
        if not chunks:
            raise HTTPException(status_code=400, detail="Unable to create chunks from extracted text.")
        
        # Create embeddings and update conversation context
        await rag_utils.create_embeddings(chunks)
        rag_utils.update_conversation_context(document_name=file.filename)

        return {
            "status": "success",
            "message": "File ingested and indexed for RAG.",
            "filename": file.filename,
            "file_size": size_bytes,
            "text_length": len(extracted_text),
            "chunks_created": len(chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/ask")
async def ask(req: AskRequest):
    """
    Answers using:
    - Top-k chunks from FAISS (if any uploaded)
    - Gemini for final, context-aware response
    """
    try:
        q = (req.question or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        answer, confidence = await rag_utils.answer_question(q)
        return {
            "status": "success",
            "question": q,
            "answer": answer,
            "confidence": round(confidence, 3),
            "timestamp": rag_utils.get_current_timestamp(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask failed: {str(e)}")

@app.post("/chat")
async def chat_message(req: AskRequest):
    """Enhanced chat endpoint with query expansion and confidence-based fallbacks"""
    try:
        # Trim and validate incoming question
        question = req.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        print(f"💬 Chat request received: '{question}'")
        
        # Step 1: Expand/rewrite the query for better retrieval
        expanded_query = await rag_utils.expand_query(question)
        print(f"🔍 Query expanded: '{question}' -> '{expanded_query}'")
        
        # Step 2: Attempt to answer using RAG with confidence scoring
        rag_answer, confidence_score = await rag_utils.answer_question(expanded_query)
        print(f"📊 RAG answer generated with confidence: {confidence_score:.3f}")
        
        # Step 3: Check if we need fallback (low confidence or missing answer)
        bot_response = rag_answer
        fallback_used = False
        
        if confidence_score < 0.5 or not rag_answer or rag_answer.strip() == "":
            print(f"⚠️ Low confidence ({confidence_score:.3f}) or missing answer, using fallback")
            
            # Use Gemini directly as fallback with conversational tone
            fallback_prompt = f"""You are a helpful AI tutor. Answer this question in a friendly, conversational style.

Question: {question}

Provide a helpful, educational response in natural conversational language. Do not use bullet points unless specifically requested."""
            
            fallback_response = await rag_utils.call_gemini(fallback_prompt)
            if fallback_response and fallback_response.lower() != "error":
                bot_response = fallback_response
                fallback_used = True
                print(f"✅ Fallback response generated successfully")
            else:
                print(f"⚠️ Fallback also failed, keeping RAG response")
        
        # Step 4: Log intermediate steps
        print(f"📝 Final response generated:")
        print(f"   - Original question: {question}")
        print(f"   - Expanded query: {expanded_query}")
        print(f"   - RAG confidence: {confidence_score:.3f}")
        print(f"   - Fallback used: {fallback_used}")
        print(f"   - Response length: {len(bot_response)}")
        
        # Step 5: Return enhanced response
        return {
            "status": "success",
            "user_message": question,
            "expanded_query": expanded_query,
            "bot_response": bot_response,
            "confidence": round(confidence_score, 3),
            "fallback_used": fallback_used,
            "timestamp": rag_utils.get_current_timestamp()
        }
    
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat message: {str(e)}")

# -------- NEW: Resource Recommendation Endpoints --------

@app.post("/api/recommend")
async def recommend_resources(req: ResourceRecommendationRequest):
    """Get comprehensive resource recommendations for a topic"""
    try:
        topic = req.topic.strip()
        if not topic:
            raise HTTPException(status_code=400, detail="Topic cannot be empty")
        
        print(f"🎯 Resource recommendation requested for: '{topic}'")
        
        results = {}
        
        # Fetch YouTube videos if requested
        if req.include_youtube:
            most_viewed = fetch_most_viewed_videos(topic)
            top_rated = fetch_top_rated_videos(topic)
            results["youtube"] = {
                "most_viewed": most_viewed,
                "top_rated": top_rated
            }
            print(f"📺 YouTube: {len(most_viewed)} most viewed, {len(top_rated)} top rated videos")
        
        # Fetch PDF resources if requested
        if req.include_pdfs:
            pdf_resources = fetch_pdf_resources(topic)
            results["pdfs"] = pdf_resources
            print(f"📄 PDFs: {len(pdf_resources)} resources found")
        
        # Fetch web resources if requested
        if req.include_web:
            web_resources = fetch_web_resources(topic)
            results["web"] = web_resources
            print(f"🌐 Web: {len(web_resources)} resources found")
        
        return {
            "status": "success",
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "resources": results,
            "summary": {
                "total_videos": len(results.get("youtube", {}).get("most_viewed", [])) + len(results.get("youtube", {}).get("top_rated", [])),
                "total_pdfs": len(results.get("pdfs", [])),
                "total_web": len(results.get("web", []))
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Resource recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Resource recommendation failed: {str(e)}")

@app.get("/api/recommend/{topic}")
async def recommend_resources_get(topic: str):
    """Get resource recommendations via GET request"""
    try:
        if not topic or topic.strip() == "":
            raise HTTPException(status_code=400, detail="Topic is required")
        
        topic = topic.strip()
        print(f"🎯 GET Resource recommendation for: '{topic}'")
        
        # Fetch all types of resources
        most_viewed = fetch_most_viewed_videos(topic)
        top_rated = fetch_top_rated_videos(topic)
        pdf_resources = fetch_pdf_resources(topic)
        web_resources = fetch_web_resources(topic)
        
        return {
            "status": "success",
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "resources": {
                "youtube": {
                    "most_viewed": most_viewed,
                    "top_rated": top_rated
                },
                "pdfs": pdf_resources,
                "web": web_resources
            },
            "summary": {
                "total_videos": len(most_viewed) + len(top_rated),
                "total_pdfs": len(pdf_resources),
                "total_web": len(web_resources)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ GET Resource recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Resource recommendation failed: {str(e)}")

@app.get("/api/youtube/{query}")
async def get_youtube_videos(query: str):
    """Get YouTube videos for a specific query"""
    try:
        if not query or query.strip() == "":
            raise HTTPException(status_code=400, detail="Query is required")
        
        query = query.strip()
        print(f"📺 YouTube search for: '{query}'")
        
        most_viewed = fetch_most_viewed_videos(query)
        top_rated = fetch_top_rated_videos(query)
        
        return {
            "status": "success",
            "query": query,
            "videos": {
                "most_viewed": most_viewed,
                "top_rated": top_rated
            },
            "total_count": len(most_viewed) + len(top_rated)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ YouTube fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"YouTube fetch failed: {str(e)}")

@app.get("/api/pdfs/{query}")
async def get_pdf_resources(query: str):
    """Get PDF resources for a specific query"""
    try:
        if not query or query.strip() == "":
            raise HTTPException(status_code=400, detail="Query is required")
        
        query = query.strip()
        print(f"📄 PDF search for: '{query}'")
        
        pdf_resources = fetch_pdf_resources(query)
        
        return {
            "status": "success",
            "query": query,
            "pdfs": pdf_resources,
            "total_count": len(pdf_resources)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ PDF resources error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF resources fetch failed: {str(e)}")

@app.post("/api/search")
async def search_resources(req: SearchRequest):
    """Search across all resource types with customizable filters"""
    try:
        query = req.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Search query is required")
        
        resource_types = req.types
        print(f"🔍 Resource search for: '{query}' (types: {resource_types})")
        
        results = {}
        
        if "youtube" in resource_types:
            results["youtube"] = {
                "most_viewed": fetch_most_viewed_videos(query),
                "top_rated": fetch_top_rated_videos(query)
            }
        
        if "pdfs" in resource_types:
            results["pdfs"] = fetch_pdf_resources(query)
        
        if "web" in resource_types:
            results["web"] = fetch_web_resources(query)
        
        return {
            "status": "success",
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Resource search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/status")
async def api_status():
    """Get comprehensive API status and configuration"""
    return {
        "status": "operational",
        "service": "AI Tutor Backend with Resource Recommendation",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "rag_system": True,
            "resource_recommendation": True,
            "youtube_integration": True,
            "pdf_resources": True,
            "web_resources": True,
            "chatbot": True
        },
        "api_keys": {
            "youtube": "configured" if YOUTUBE_API_KEY else "missing",
            "gemini": "configured" if os.getenv("GEMINI_API_KEY") else "missing"
        },
        "endpoints": [
            "GET /",
            "GET /demo",
            "GET /health",
            "POST /upload-file",
            "POST /ask",
            "POST /chat",
            "POST /api/recommend",
            "GET /api/recommend/{topic}",
            "GET /api/youtube/{query}",
            "GET /api/pdfs/{query}",
            "POST /api/search",
            "GET /api/status",
            "GET /status",
            "GET /conversation-context",
            "POST /clear-context"
        ]
    }

# -------- Existing Routes --------

@app.get("/status")
async def status():
    try:
        return {
            "status": "success",
            "index_initialized": rag_utils.is_index_initialized(),
            "chunks_count": rag_utils.get_chunks_count(),
            "embeddings_count": rag_utils.get_embeddings_count(),
            "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status failed: {str(e)}")

@app.get("/conversation-context")
async def get_conversation_context():
    """Get current conversation context for debugging"""
    try:
        context = rag_utils.get_conversation_context()
        return {
            "status": "success",
            "conversation_context": context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting conversation context: {str(e)}")

@app.get("/current-document")
async def get_current_document():
    """Get current document name for the frontend"""
    try:
        context = rag_utils.get_conversation_context()
        return {
            "status": "success",
            "current_document": context.get("current_document"),
            "has_document": bool(context.get("current_document"))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting current document: {str(e)}")

@app.post("/clear-context")
async def clear_conversation_context():
    """Clear the current conversation context"""
    try:
        rag_utils.clear_conversation_context()
        return {
            "status": "success",
            "message": "Conversation context cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing conversation context: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Tutor Backend with Resource Recommendation...")
    print("📱 RAG System: Active")
    print("💬 Chatbot: Active")
    print("📺 YouTube Integration: Active")
    print("📄 PDF Resources: Active")
    print("🌐 Web Resources: Active")
    print("🎨 Demo Frontend: Active")
    print("🔧 All endpoints ready for Postman testing!")
    print("🌍 Server running on: http://localhost:8001")
    print("🎨 Frontend available at: http://localhost:8001/")
    print("\n📋 Available endpoints:")
    print("   GET  / (Frontend)")
    print("   GET  /demo (Frontend)")
    print("   GET  /health")
    print("   POST /upload-file")
    print("   POST /ask")
    print("   POST /chat")
    print("   POST /api/recommend")
    print("   GET  /api/recommend/{topic}")
    print("   GET  /api/youtube/{query}")
    print("   GET  /api/pdfs/{query}")
    print("   POST /api/search")
    print("   GET  /api/status")
    print("   GET  /status")
    print("   GET  /conversation-context")
    print("   POST /clear-context")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
