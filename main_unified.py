from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import requests
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = './pdfs'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'doc', 'rtf', 'odt', 'md'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# API Keys (replace with your actual keys)
YOUTUBE_API_KEY = "AIzaSyAIOx5R8O1oAv97rH0-8W8i5ObI_X0801M"
GEMINI_API_KEY = "AIzaSyDUIDfqOnL4NNgO63pYE5UMG35-uRKQWq8"  # Add your Gemini API key

# ==================== RESOURCE RECOMMENDATION SYSTEM ====================

def fetch_most_viewed_videos(query):
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

def fetch_top_rated_videos(query):
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

def fetch_pdf_resources(query):
    """Fetch PDF resources for a topic"""
    try:
        # Enhanced PDF search with multiple sources
        pdf_sources = [
            {
                "title": f"Google Search: {query} PDFs",
                "link": f"https://www.google.com/search?q={query}+filetype:pdf",
                "source": "Google Search"
            },
            {
                "title": f"Scholar Search: {query} Academic Papers",
                "link": f"https://scholar.google.com/scholar?q={query}",
                "source": "Google Scholar"
            },
            {
                "title": f"ResearchGate: {query} Research Papers",
                "link": f"https://www.researchgate.net/search/publication?q={query}",
                "source": "ResearchGate"
            }
        ]
        return pdf_sources
    except Exception as e:
        print(f"Error fetching PDF resources: {e}")
        return []

def fetch_web_resources(query):
    """Fetch general web resources for a topic"""
    try:
        # You can enhance this with actual web scraping or API calls
        web_sources = [
            {
                "title": f"Wikipedia: {query}",
                "link": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                "source": "Wikipedia"
            },
            {
                "title": f"Stack Overflow: {query} Programming Questions",
                "link": f"https://stackoverflow.com/search?q={query}",
                "source": "Stack Overflow"
            }
        ]
        return web_sources
    except Exception as e:
        print(f"Error fetching web resources: {e}")
        return []

# ==================== MAIN API ENDPOINTS ====================

@app.route('/')
def home():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Unified AI Tutor Backend',
        'timestamp': datetime.now().isoformat(),
        'features': [
            'Resource Recommendation',
            'PDF Processing',
            'Study Materials',
            'Quiz Generation'
        ]
    })

@app.route('/api/recommend', methods=['POST'])
def recommend_resources():
    """Get resource recommendations for a topic"""
    try:
        data = request.get_json()
        if not data or 'topic' not in data:
            return jsonify({'error': 'Topic is required'}), 400
        
        topic = data['topic'].strip()
        if not topic:
            return jsonify({'error': 'Topic cannot be empty'}), 400
        
        # Fetch all types of resources
        most_viewed = fetch_most_viewed_videos(topic)
        top_rated = fetch_top_rated_videos(topic)
        pdf_resources = fetch_pdf_resources(topic)
        web_resources = fetch_web_resources(topic)
        
        return jsonify({
            'success': True,
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'resources': {
                'youtube': {
                    'most_viewed': most_viewed,
                    'top_rated': top_rated
                },
                'pdfs': pdf_resources,
                'web': web_resources
            },
            'summary': {
                'total_videos': len(most_viewed) + len(top_rated),
                'total_pdfs': len(pdf_resources),
                'total_web': len(web_resources)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Recommendation failed: {str(e)}'}), 500

@app.route('/api/recommend/<topic>', methods=['GET'])
def recommend_resources_get(topic):
    """Get resource recommendations via GET request"""
    try:
        if not topic or topic.strip() == '':
            return jsonify({'error': 'Topic is required'}), 400
        
        topic = topic.strip()
        
        # Fetch all types of resources
        most_viewed = fetch_most_viewed_videos(topic)
        top_rated = fetch_top_rated_videos(topic)
        pdf_resources = fetch_pdf_resources(topic)
        web_resources = fetch_web_resources(topic)
        
        return jsonify({
            'success': True,
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'resources': {
                'youtube': {
                    'most_viewed': most_viewed,
                    'top_rated': top_rated
                },
                'pdfs': pdf_resources,
                'web': web_resources
            },
            'summary': {
                'total_videos': len(most_viewed) + len(top_rated),
                'total_pdfs': len(pdf_resources),
                'total_web': len(web_resources)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Recommendation failed: {str(e)}'}), 500

@app.route('/api/youtube/<query>', methods=['GET'])
def get_youtube_videos(query):
    """Get YouTube videos for a specific query"""
    try:
        if not query or query.strip() == '':
            return jsonify({'error': 'Query is required'}), 400
        
        query = query.strip()
        most_viewed = fetch_most_viewed_videos(query)
        top_rated = fetch_top_rated_videos(query)
        
        return jsonify({
            'success': True,
            'query': query,
            'videos': {
                'most_viewed': most_viewed,
                'top_rated': top_rated
            },
            'total_count': len(most_viewed) + len(top_rated)
        })
        
    except Exception as e:
        return jsonify({'error': f'YouTube fetch failed: {str(e)}'}), 500

@app.route('/api/pdfs/<query>', methods=['GET'])
def get_pdf_resources(query):
    """Get PDF resources for a specific query"""
    try:
        if not query or query.strip() == '':
            return jsonify({'error': 'Query is required'}), 400
        
        query = query.strip()
        pdf_resources = fetch_pdf_resources(query)
        
        return jsonify({
            'success': True,
            'query': query,
            'pdfs': pdf_resources,
            'total_count': len(pdf_resources)
        })
        
    except Exception as e:
        return jsonify({'error': f'PDF resources fetch failed: {str(e)}'}), 500

@app.route('/api/search', methods=['POST'])
def search_resources():
    """Search across all resource types"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Search query is required'}), 400
        
        query = data['query'].strip()
        resource_types = data.get('types', ['youtube', 'pdfs', 'web'])
        
        results = {}
        
        if 'youtube' in resource_types:
            results['youtube'] = {
                'most_viewed': fetch_most_viewed_videos(query),
                'top_rated': fetch_top_rated_videos(query)
            }
        
        if 'pdfs' in resource_types:
            results['pdfs'] = fetch_pdf_resources(query)
        
        if 'web' in resource_types:
            results['web'] = fetch_web_resources(query)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

@app.route('/api/status', methods=['GET'])
def system_status():
    """Get system status and configuration"""
    return jsonify({
        'status': 'operational',
        'service': 'Unified AI Tutor Backend',
        'timestamp': datetime.now().isoformat(),
        'features': {
            'resource_recommendation': True,
            'youtube_integration': True,
            'pdf_resources': True,
            'web_resources': True
        },
        'api_keys': {
            'youtube': 'configured' if YOUTUBE_API_KEY else 'missing',
            'gemini': 'configured' if GEMINI_API_KEY != 'your_gemini_api_key_here' else 'missing'
        },
        'endpoints': [
            'GET /health',
            'POST /api/recommend',
            'GET /api/recommend/<topic>',
            'GET /api/youtube/<query>',
            'GET /api/pdfs/<query>',
            'POST /api/search',
            'GET /api/status'
        ]
    })

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    print("🚀 Starting Unified AI Tutor Backend...")
    print("📱 Resource Recommendation System: Active")
    print("📺 YouTube Integration: Active")
    print("📄 PDF Resources: Active")
    print("🌐 Web Resources: Active")
    print("🔧 API Endpoints: Ready for Postman testing")
    print("🌍 Server running on: http://localhost:5000")
    print("\n📋 Available endpoints:")
    print("   GET  /health")
    print("   POST /api/recommend")
    print("   GET  /api/recommend/<topic>")
    print("   GET  /api/youtube/<query>")
    print("   GET  /api/pdfs/<query>")
    print("   POST /api/search")
    print("   GET  /api/status")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
