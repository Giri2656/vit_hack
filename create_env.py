#!/usr/bin/env python3
"""
Script to create .env file with correct configuration
"""

import os

def create_env_file():
    """Create .env file with default configuration"""
    
    env_content = """# Gemini API Configuration
GEMINI_API_KEY=AIzaSyAnpr0uJyTDBhZx_3pONlzcp1reiCL-Za4

# Server Configuration
PORT=8000
FAISS_DIMENSION=768

# File Upload Configuration
ALLOWED_FILE_TYPES=*/*
MAX_FILE_SIZE=10485760

# Logging
LOG_LEVEL=info
"""
    
    # Write .env file
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ .env file created successfully!")
    print("📁 File location:", os.path.abspath('.env'))
    print("🔧 Configuration:")
    print("   - ALLOWED_FILE_TYPES: */* (accepts all files)")
    print("   - MAX_FILE_SIZE: 10MB")
    print("   - GEMINI_API_KEY: Configured")

if __name__ == "__main__":
    create_env_file()
