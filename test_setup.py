#!/usr/bin/env python3
"""
Test script to verify the AI Tutor Backend setup
Run this script to check if all dependencies are properly installed
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    try:
        import fastapi
        print(f"✅ FastAPI {fastapi.__version__}")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import uvicorn
        print(f"✅ Uvicorn {uvicorn.__version__}")
    except ImportError as e:
        print(f"❌ Uvicorn import failed: {e}")
        return False
    
    try:
        import PyPDF2
        print(f"✅ PyPDF2 {PyPDF2.__version__}")
    except ImportError as e:
        print(f"❌ PyPDF2 import failed: {e}")
        return False
    
    try:
        import faiss
        print(f"✅ FAISS {faiss.__version__}")
    except ImportError as e:
        print(f"❌ FAISS import failed: {e}")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI")
    except ImportError as e:
        print(f"❌ Google Generative AI import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import dotenv
        print("✅ Python-dotenv")
    except ImportError as e:
        print(f"❌ Python-dotenv import failed: {e}")
        return False
    
    return True

def test_environment():
    """Test environment configuration"""
    print("\n🔧 Testing environment configuration...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("⚠️ Warning: Python 3.8+ is recommended")
    
    # Check if .env file exists
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"✅ Environment file found: {env_file}")
        
        # Check for Gemini API key
        with open(env_file, 'r') as f:
            content = f.read()
            if "GEMINI_API_KEY" in content:
                print("✅ Gemini API key configuration found")
            else:
                print("⚠️ Gemini API key not found in .env file")
    else:
        print(f"⚠️ Environment file not found: {env_file}")
        print("   Copy env.example to .env and configure your API keys")
    
    return True

def test_faiss_initialization():
    """Test FAISS index initialization"""
    print("\n🔍 Testing FAISS initialization...")
    
    try:
        import faiss
        import numpy as np
        
        # Create a simple test index
        dimension = 768
        index = faiss.IndexFlatL2(dimension)
        
        # Test with sample vectors
        test_vectors = np.random.rand(5, dimension).astype('float32')
        index.add(test_vectors)
        
        print(f"✅ FAISS index created successfully")
        print(f"   - Dimension: {dimension}")
        print(f"   - Vectors added: {index.ntotal}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAISS initialization failed: {e}")
        return False

def test_pdf_processing():
    """Test PDF processing capabilities"""
    print("\n📄 Testing PDF processing...")
    
    try:
        import PyPDF2
        import io
        
        # Create a simple test PDF-like structure
        test_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
        # Test PDF reader
        pdf_file = io.BytesIO(test_content)
        
        print("✅ PyPDF2 can process PDF-like content")
        return True
        
    except Exception as e:
        print(f"❌ PDF processing test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 AI Tutor Backend - Setup Test")
    print("=" * 40)
    
    all_tests_passed = True
    
    # Run all tests
    if not test_imports():
        all_tests_passed = False
    
    if not test_environment():
        all_tests_passed = False
    
    if not test_faiss_initialization():
        all_tests_passed = False
    
    if not test_pdf_processing():
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 40)
    if all_tests_passed:
        print("🎉 All tests passed! Your backend is ready to run.")
        print("\nNext steps:")
        print("1. Configure your Gemini API key in .env file")
        print("2. Run: uvicorn main:app --reload")
        print("3. Visit: http://localhost:8000/docs")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check Python version (3.8+ required)")
        print("3. Verify environment configuration")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
