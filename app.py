import os
from flask import Flask, request, jsonify
import vertexai
from vertexai.generative_models import GenerativeModel
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Vertex AI setup
vertexai.init(project=os.getenv('GCP_PROJECT_ID'), location='us-central1')
embeddings = VertexAIEmbeddings(model_name='text-embedding-004')
vectorstore_path = '/app/db/chroma'
vectorstore = None

def get_db_connection():
    db_user = os.getenv('DB_USER')
    db_pass = os.getenv('DB_PASS')
    db_host = os.getenv('DB_HOST')
    db_name = os.getenv('DB_NAME')
    engine = create_engine(f'mysql+pymysql://{db_user}:{db_pass}@{db_host}/{db_name}')
    return engine

def init_vectorstore():
    global vectorstore
    if not vectorstore:
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/ingest', methods=['POST'])
def ingest_doc():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    loader = PyPDFLoader(file)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    init_vectorstore()
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    
    engine = get_db_connection()
    with engine.connect() as conn:
        conn.execute(text("INSERT INTO docs (name) VALUES (:name)"), {'name': file.filename})
        conn.commit()
    
    return jsonify({'status': 'ingested'})

@app.route('/query', methods=['POST'])
def query_rag():
    query_text = request.json.get('query')
    init_vectorstore()
    docs = vectorstore.similarity_search(query_text, k=3)
    context = '\n'.join([doc.page_content for doc in docs])
    
    model = GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(f"Context: {context}\nQuestion: {query_text}")
    
    return jsonify({'answer': response.text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)