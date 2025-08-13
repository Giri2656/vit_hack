#!/usr/bin/env python3
"""
Simple test script to verify file upload and question answering
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Health check: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_status():
    """Test status endpoint"""
    print("\n📊 Testing status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        print(f"✅ Status check: {response.status_code}")
        data = response.json()
        print(f"   Embeddings: {data.get('embeddings_count', 0)}")
        print(f"   Chunks: {data.get('chunks_count', 0)}")
        print(f"   Index initialized: {data.get('index_initialized', False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False

def test_documents():
    """Test documents endpoint"""
    print("\n📚 Testing documents endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/documents")
        print(f"✅ Documents check: {response.status_code}")
        data = response.json()
        print(f"   Documents count: {data.get('documents_count', 0)}")
        print(f"   FAISS vectors: {data.get('faiss_vectors', 0)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Documents check failed: {e}")
        return False

def test_question_without_docs():
    """Test asking a question without documents"""
    print("\n❓ Testing question without documents...")
    try:
        question = "What is machine learning?"
        response = requests.post(f"{BASE_URL}/ask", json={"question": question})
        print(f"✅ Question asked: {response.status_code}")
        data = response.json()
        print(f"   Answer preview: {data.get('answer', '')[:100]}...")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Question failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 AI Tutor Backend Test Suite")
    print("=" * 40)
    
    # Check if server is running
    if not test_health():
        print("\n❌ Server is not running. Please start the server first:")
        print("   cd backend && python main.py")
        return
    
    # Test status
    test_status()
    
    # Test documents (should be empty initially)
    test_documents()
    
    # Test question without documents
    test_question_without_docs()
    
    print("\n" + "=" * 40)
    print("✅ Basic tests completed!")
    print("\nNext steps:")
    print("1. Upload a file using POST /upload-file")
    print("2. Check /documents to see if it was stored")
    print("3. Ask a question about the uploaded content")

if __name__ == "__main__":
    main()
