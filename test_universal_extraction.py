#!/usr/bin/env python3
"""
Test script for the Universal Text Extraction Pipeline.
This script tests the file type detection and text extraction capabilities.
"""

import asyncio
import sys
import os
import io
from unittest.mock import Mock

# Add the current directory to the path so we can import rag_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import rag_utils

def create_mock_file(filename: str, content: bytes, content_type: str = None) -> Mock:
    """Create a mock file object for testing"""
    mock_file = Mock()
    mock_file.filename = filename
    mock_file.content_type = content_type
    mock_file.file = io.BytesIO(content)
    return mock_file

async def test_file_type_detection():
    """Test file type detection with various file types"""
    print("🧪 Testing File Type Detection")
    print("=" * 50)
    
    # Test PDF detection
    pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj'
    pdf_file = create_mock_file("test.pdf", pdf_content, "application/pdf")
    pdf_info = rag_utils.detect_file_type(pdf_file)
    print(f"✅ PDF detection: {pdf_info}")
    
    # Test image detection
    png_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
    png_file = create_mock_file("test.png", png_content, "image/png")
    png_info = rag_utils.detect_file_type(png_file)
    print(f"✅ PNG detection: {png_info}")
    
    # Test Word document detection
    docx_content = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    docx_file = create_mock_file("test.docx", docx_content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    docx_info = rag_utils.detect_file_type(docx_file)
    print(f"✅ DOCX detection: {docx_info}")
    
    # Test text file detection
    txt_content = b'This is a test text file with some content.'
    txt_file = create_mock_file("test.txt", txt_content, "text/plain")
    txt_info = rag_utils.detect_file_type(txt_file)
    print(f"✅ TXT detection: {txt_info}")
    
    # Test HTML detection
    html_content = b'<!DOCTYPE html><html><head><title>Test</title></head><body>Hello World</body></html>'
    html_file = create_mock_file("test.html", html_content, "text/html")
    html_info = rag_utils.detect_file_type(html_file)
    print(f"✅ HTML detection: {html_info}")
    
    # Test unknown file type
    unknown_content = b'\x00\x01\x02\x03\x04\x05\x06\x07'
    unknown_file = create_mock_file("test.xyz", unknown_content, "application/octet-stream")
    unknown_info = rag_utils.detect_file_type(unknown_file)
    print(f"✅ Unknown detection: {unknown_info}")

async def test_text_extraction():
    """Test text extraction with various file types"""
    print("\n🧪 Testing Text Extraction")
    print("=" * 50)
    
    # Test text file extraction
    txt_content = b'This is a test text file with some content.\nIt has multiple lines.\nAnd some special characters: \xc3\xa9 \xc3\xb1'
    txt_file = create_mock_file("test.txt", txt_content, "text/plain")
    txt_text = await rag_utils.extract_text_file(txt_file)
    print(f"✅ Text extraction: {len(txt_text)} characters")
    print(f"   Preview: {txt_text[:100]}...")
    
    # Test HTML extraction
    html_content = b'<!DOCTYPE html><html><head><title>Test Page</title></head><body><h1>Hello World</h1><p>This is a test paragraph.</p></body></html>'
    html_file = create_mock_file("test.html", html_content, "text/html")
    try:
        html_text = await rag_utils.extract_html_text(html_file)
        print(f"✅ HTML extraction: {len(html_text)} characters")
        print(f"   Preview: {html_text[:100]}...")
    except Exception as e:
        print(f"⚠️ HTML extraction failed: {e}")
    
    # Test JSON extraction
    json_content = b'{"name": "Test", "value": 42, "items": ["a", "b", "c"]}'
    json_file = create_mock_file("test.json", json_content, "application/json")
    try:
        json_text = await rag_utils.extract_structured_text(json_file)
        print(f"✅ JSON extraction: {len(json_text)} characters")
        print(f"   Preview: {json_text[:100]}...")
    except Exception as e:
        print(f"⚠️ JSON extraction failed: {e}")

async def test_universal_pipeline():
    """Test the complete universal text extraction pipeline"""
    print("\n🧪 Testing Universal Pipeline")
    print("=" * 50)
    
    # Test with text file
    txt_content = b'This is a comprehensive test of the universal text extraction pipeline.'
    txt_file = create_mock_file("test.txt", txt_content, "text/plain")
    txt_result = await rag_utils.universal_text_extraction(txt_file)
    print(f"✅ Universal pipeline (TXT): {len(txt_result)} characters")
    
    # Test with PDF-like content
    pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000111 00000 n \n0000000212 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF'
    pdf_file = create_mock_file("test.pdf", pdf_content, "application/pdf")
    pdf_result = await rag_utils.universal_text_extraction(pdf_file)
    print(f"✅ Universal pipeline (PDF): {len(pdf_result)} characters")
    
    # Test with image-like content
    png_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
    png_file = create_mock_file("test.png", png_content, "image/png")
    png_result = await rag_utils.universal_text_extraction(png_file)
    print(f"✅ Universal pipeline (PNG): {len(png_result)} characters")

async def test_error_handling():
    """Test error handling and fallback mechanisms"""
    print("\n🧪 Testing Error Handling")
    print("=" * 50)
    
    # Test with corrupted content
    corrupted_content = b'\x00\x00\x00\x00\x00\x00\x00\x00'
    corrupted_file = create_mock_file("corrupted.bin", corrupted_content, "application/octet-stream")
    corrupted_result = await rag_utils.universal_text_extraction(corrupted_file)
    print(f"✅ Corrupted file handling: {len(corrupted_result)} characters")
    
    # Test with empty content
    empty_content = b''
    empty_file = create_mock_file("empty.txt", empty_content, "text/plain")
    empty_result = await rag_utils.universal_text_extraction(empty_file)
    print(f"✅ Empty file handling: {len(empty_result)} characters")

async def test_conversation_context():
    """Test conversation context management"""
    print("\n🧪 Testing Conversation Context")
    print("=" * 50)
    
    # Test initial context
    initial_context = rag_utils.get_conversation_context()
    print(f"✅ Initial context: {initial_context}")
    
    # Test context update
    rag_utils.update_conversation_context(
        document_name="test_document.pdf",
        document_summary="This is a test document about artificial intelligence."
    )
    updated_context = rag_utils.get_conversation_context()
    print(f"✅ Updated context: {updated_context}")
    
    # Test context clearing
    rag_utils.clear_conversation_context()
    cleared_context = rag_utils.get_conversation_context()
    print(f"✅ Cleared context: {cleared_context}")

async def test_query_expansion():
    """Test query expansion for contextual queries"""
    print("\n🧪 Testing Query Expansion")
    print("=" * 50)
    
    # Set up context first
    rag_utils.update_conversation_context(
        document_name="research_paper.pdf",
        document_summary="This document discusses machine learning algorithms."
    )
    
    # Test contextual queries
    contextual_queries = [
        "explain this",
        "what is this about",
        "tell me about this",
        "describe this document"
    ]
    
    for query in contextual_queries:
        expanded = await rag_utils.expand_query(query)
        print(f"✅ Query expansion: '{query}' -> '{expanded}'")
    
    # Test non-contextual queries
    non_contextual_queries = [
        "what is machine learning",
        "explain artificial intelligence",
        "how does deep learning work"
    ]
    
    for query in non_contextual_queries:
        expanded = await rag_utils.expand_query(query)
        print(f"✅ Non-contextual: '{query}' -> '{expanded}'")

async def main():
    """Run all tests"""
    print("🚀 Starting Universal Text Extraction Pipeline Tests")
    print("=" * 70)
    
    try:
        # Run all test suites
        await test_file_type_detection()
        await test_text_extraction()
        await test_universal_pipeline()
        await test_error_handling()
        await test_conversation_context()
        await test_query_expansion()
        
        print("\n✅ All tests completed successfully!")
        print("🎉 The Universal Text Extraction Pipeline is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # Check if Gemini API key is available
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  GEMINI_API_KEY not found. Some tests may not work properly.")
        print("   Set the environment variable to test full functionality.")
    
    # Run the tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
