from flask import Flask, request, jsonify
import os
import mysql.connector
from google.cloud import secretmanager
from google.cloud import aiplatform
import chromadb
import logging
from datetime import datetime
import json
import uuid

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Vertex AI
aiplatform.init(project="sabre-ai", location="europe-west4")

# Secret Manager client
def get_secret(secret_id):
    """Retrieve secret from Secret Manager"""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/sabre-ai/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.error(f"Error accessing secret {secret_id}: {str(e)}")
        return None

# Database connection
def get_db_connection():
    """Get database connection using secrets"""
    db_password = get_secret("db-password")
    
    if not db_password:
        logger.error("Failed to retrieve database password")
        return None
    
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST', '127.0.0.1'),
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER', 'root'),
            password=db_password,
            database=os.getenv('DB_NAME', 'sabre_ai_db')
        )
        return connection
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return None

# ChromaDB setup
def get_chroma_client():
    """Get ChromaDB client"""
    try:
        client = chromadb.PersistentClient(path=os.getenv('CHROMA_PATH', '/tmp/chroma'))
        return client
    except Exception as e:
        logger.error(f"ChromaDB initialization failed: {str(e)}")
        return None

# Vertex AI embedding
def get_embedding(text):
    """Get embedding using Vertex AI"""
    try:
        from vertexai.language_models import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
        embeddings = model.get_embeddings([text])
        return embeddings[0].values
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "sabre-ai"
    })

@app.route('/api/test-db', methods=['GET'])
def test_database():
    """Test database connection"""
    connection = get_db_connection()
    
    if not connection:
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT DATABASE() as db_name")
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        return jsonify({
            "status": "success",
            "database": result[0],
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Database query failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/test-embedding', methods=['POST'])
def test_embedding():
    """Test Vertex AI embedding"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400
    
    text = data['text']
    embedding = get_embedding(text)
    
    if not embedding:
        return jsonify({"error": "Embedding generation failed"}), 500
    
    return jsonify({
        "status": "success",
        "text": text,
        "embedding_length": len(embedding),
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/api/test-chroma', methods=['GET'])
def test_chroma():
    """Test ChromaDB connection"""
    client = get_chroma_client()
    
    if not client:
        return jsonify({"error": "ChromaDB connection failed"}), 500
    
    try:
        collection = client.get_or_create_collection(name="test_collection")
        test_id = str(uuid.uuid4())
        collection.add(
            documents=["test document"],
            embeddings=[[0.1] * 768],  # Simple dummy embedding
            ids=[test_id]
        )
        
        results = collection.query(
            query_embeddings=[[0.1] * 768],
            n_results=1
        )
        
        collection.delete(ids=[test_id])
        
        return jsonify({
            "status": "success",
            "collection_name": "test_collection",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"ChromaDB test failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ingest-document', methods=['POST'])
def ingest_document():
    """Ingest document into ChromaDB"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400
    
    text = data['text']
    doc_id = str(uuid.uuid4())
    
    # Generate embedding
    embedding = get_embedding(text)
    if not embedding:
        return jsonify({"error": "Failed to generate embedding"}), 500
    
    # Store in ChromaDB
    client = get_chroma_client()
    if not client:
        return jsonify({"error": "ChromaDB connection failed"}), 500
    
    try:
        collection = client.get_or_create_collection(name="documents")
        
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id]
        )
        
        return jsonify({
            "status": "success",
            "document_id": doc_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_documents():
    """Search documents using semantic similarity"""
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request"}), 400
    
    query_text = data['query']
    query_embedding = get_embedding(query_text)
    
    if not query_embedding:
        return jsonify({"error": "Failed to generate query embedding"}), 500
    
    client = get_chroma_client()
    if not client:
        return jsonify({"error": "ChromaDB connection failed"}), 500
    
    try:
        collection = client.get_collection(name="documents")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=['documents', 'distances']
        )
        
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                search_results.append({
                    "content": doc,
                    "distance": results['distances'][0][i] if i < len(results['distances'][0]) else 0.0,
                    "rank": i + 1
                })
        
        return jsonify({
            "status": "success",
            "query": query_text,
            "results": search_results,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        "message": "Sabre AI - RAG System",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "test-db": "/api/test-db",
            "test-embedding": "/api/test-embedding",
            "test-chroma": "/api/test-chroma",
            "ingest-document": "/api/ingest-document",
            "search": "/api/search"
        },
        "timestamp": datetime.utcnow().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
