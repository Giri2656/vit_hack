#!/usr/bin/env python3
"""
Startup script for AI Tutor Backend
This script provides an easy way to start the FastAPI server
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

def main():
    """Main startup function"""
    print("🚀 Starting AI Tutor Backend...")
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("NODE_ENV", "development") == "development"
    
    # Check if Gemini API key is configured
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("⚠️ Warning: GEMINI_API_KEY not found in environment")
        print("   The system will use fallback embeddings (random vectors)")
        print("   For full functionality, set GEMINI_API_KEY in your .env file")
    else:
        print("✅ Gemini API key configured")
    
    print(f"🌐 Server will start on {host}:{port}")
    print(f"🔄 Auto-reload: {'enabled' if reload else 'disabled'}")
    print(f"📚 API docs will be available at: http://{host}:{port}/docs")
    print(f"📁 File uploads: PDF, DOC, DOCX, TXT, PNG, JPEG, JPG")
    
    try:
        # Start the server
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
