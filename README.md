# AI Tutor Platform

A comprehensive AI-powered learning platform with RAG (Retrieval-Augmented Generation), resource recommendations, and interactive chat capabilities.

## Features

### 🧠 AI-Powered Learning
- **Universal File Ingestion**: Upload PDFs, DOCs, PPTs, images, and more
- **RAG System**: Context-aware responses using document embeddings
- **Smart Chat**: Interactive AI tutor with conversation memory
- **Query Expansion**: Enhanced question understanding

### 📚 Resource Recommendations
- **YouTube Integration**: Most viewed and top-rated educational videos
- **PDF Resources**: Academic papers, research documents, and study materials
- **Web Resources**: Wikipedia, Stack Overflow, Reddit, and more
- **Dynamic Search**: Real-time resource discovery based on topics

### 🎯 Learning Tools
- **Interactive Chat**: Upload documents and ask questions
- **Resource Browser**: Discover learning materials
- **Profile Management**: Track progress and preferences
- **Quiz System**: Test knowledge and understanding
- **Learning Roadmap**: Personalized learning paths

## Quick Start

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your API keys:
# GEMINI_API_KEY=your_gemini_api_key_here

# Start the backend server
python main.py
```

The backend will be available at `http://localhost:8001`

### 2. Frontend Access

Once the backend is running, access the frontend at:
- **Main Application**: `http://localhost:8001/`
- **Login Page**: `http://localhost:8001/login.html`
- **Chat Interface**: `http://localhost:8001/chat.html`
- **Resource Browser**: `http://localhost:8001/resource.html`

### 3. First Steps

1. **Register/Login**: Create an account or sign in
2. **Upload Documents**: Use the chat interface to upload study materials
3. **Ask Questions**: Get AI-powered answers based on your documents
4. **Discover Resources**: Browse recommended learning materials
5. **Track Progress**: Monitor your learning journey in the profile section

## API Endpoints

### Core Endpoints
- `GET /` - Main frontend
- `GET /health` - Health check
- `GET /status` - System status and statistics
- `POST /upload-file` - Upload and process documents
- `POST /chat` - Interactive chat with AI
- `POST /ask` - Simple question answering

### Resource Recommendation
- `POST /api/recommend` - Get comprehensive resource recommendations
- `GET /api/recommend/{topic}` - Get resources for specific topic
- `GET /api/youtube/{query}` - YouTube video search
- `GET /api/pdfs/{query}` - PDF resource search
- `POST /api/search` - Multi-resource search

### System Management
- `GET /conversation-context` - Get current chat context
- `POST /clear-context` - Clear conversation context
- `GET /api/status` - Detailed API status

## Frontend Features

### 🔐 Authentication
- User registration and login
- Session management with localStorage
- Profile customization
- Learning preferences

### 💬 Chat Interface
- **File Upload**: Support for multiple file formats
- **Real-time Chat**: Interactive AI responses
- **Context Management**: Conversation memory
- **Resource Panel**: Related learning materials
- **Typing Indicators**: Visual feedback during processing

### 📚 Resource Browser
- **Dynamic Search**: Real-time resource discovery
- **Multiple Sources**: YouTube, PDFs, web resources
- **Filtered Results**: Curated educational content
- **Quick Actions**: Easy access to learning materials

### 👤 Profile Management
- **User Preferences**: Learning level and subject preferences
- **Statistics Tracking**: Questions asked, study sessions, time spent
- **Data Export**: Download learning data
- **System Status**: Backend connection monitoring

## File Support

The system supports various file formats for document ingestion:

### Text Documents
- PDF (.pdf)
- Microsoft Word (.doc, .docx)
- Plain Text (.txt)
- Rich Text (.rtf)

### Presentations
- PowerPoint (.ppt, .pptx)

### Spreadsheets
- Excel (.xls, .xlsx)

### Web Content
- HTML (.html, .htm)
- CSS (.css)

### Images (OCR)
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)

## Configuration

### Environment Variables
Create a `.env` file in the backend directory:

```env
# Required: Gemini API Key for AI responses
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: YouTube API Key for video recommendations
YOUTUBE_API_KEY=your_youtube_api_key_here

# Optional: Database configuration
DATABASE_URL=sqlite:///ai_tutor.db
```

### API Keys Setup

1. **Gemini API Key** (Required):
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add to your `.env` file

2. **YouTube API Key** (Optional):
   - Visit [Google Cloud Console](https://console.cloud.google.com/)
   - Enable YouTube Data API v3
   - Create credentials
   - Add to your `.env` file

## Usage Examples

### Upload and Chat
1. Go to the chat interface
2. Click "Upload File" and select a document
3. Wait for processing confirmation
4. Ask questions about the document content
5. Get AI-powered answers with context

### Resource Discovery
1. Visit the resource browser
2. Search for topics (e.g., "machine learning", "biology")
3. Browse recommended videos, PDFs, and web resources
4. Click on resources to open them in new tabs

### Profile Management
1. Access your profile page
2. Update learning preferences and subjects
3. View learning statistics
4. Export your data or clear history

## Development

### Backend Development
```bash
# Install development dependencies
pip install -r requirements_enhanced.txt

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
python test_setup.py
python test_upload.py
python test_universal_extraction.py
```

### Frontend Development
The frontend uses:
- **Tailwind CSS** for styling
- **Font Awesome** for icons
- **Fetch API** for backend communication
- **LocalStorage** for session management

### Adding New Features
1. **Backend**: Add new endpoints in `main.py`
2. **Frontend**: Create new HTML pages in `static/`
3. **Integration**: Use fetch API to connect frontend and backend
4. **Testing**: Test both API endpoints and frontend functionality

## Troubleshooting

### Common Issues

1. **Backend Connection Failed**:
   - Check if the backend server is running
   - Verify the API_BASE_URL in frontend files
   - Check firewall settings

2. **File Upload Issues**:
   - Ensure file format is supported
   - Check file size limits
   - Verify Gemini API key is configured

3. **Resource Recommendations Not Loading**:
   - Check YouTube API key configuration
   - Verify internet connection
   - Check browser console for errors

4. **Authentication Issues**:
   - Clear browser localStorage
   - Check if user data is properly stored
   - Verify login/logout flow

### Debug Mode
Enable debug logging by setting environment variables:
```env
DEBUG=true
LOG_LEVEL=DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the troubleshooting section
- Review the API documentation
- Open an issue on GitHub

---

**Happy Learning! 🎓**
