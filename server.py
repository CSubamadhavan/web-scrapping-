import re
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
import jwt as pyjwt
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import nltk
import subprocess
import os
import json
from datetime import timezone
import time
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from crawler.spider import URLCollector
from scraper.selenium_scraper import SeleniumScraper
from processor.cleaner import Cleaner
from processor.lemmatizer import TextLemmatizer
from classifier.keybert_topic_extractor import KeyBERTTopicExtractor
from classifier.topic_classifier import TopicClassifier
from question.embedding import connect_milvus, ensure_collection
from question.question_generator import QuestionGenerator
from question.question_answerer import QuestionAnswerer
from processor.contact_info_extractor import extract_contact_info
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from response import get_response_handler
import traceback
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from threading import Lock
import threading

qa_lock = threading.Lock()
# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print("[✓] NLTK resources downloaded successfully")
except Exception as e:
    print(f"[WARNING] NLTK download issue: {e}")

# Global variables for Milvus
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "webpages"
sentence_model = None
milvus_collection = None

# Global model instances for processing
topic_extractor = None
classifier = None
question_generator = None
question_answerer = None
_models_initialized = False
_initialization_lock = Lock()

def init_models():
    """Initialize all models once and only once"""
    global sentence_model, topic_extractor, classifier, question_generator, question_answerer, milvus_collection
    global _models_initialized
    # Thread-safe check to prevent multiple initializations
    with _initialization_lock:
        if _models_initialized:
            print("[INFO] Models already initialized, skipping...")
            return True
       
        try:
            print("⚙️ Initializing models...")

            # Initialize topic models with error handling
            try:
                topic_extractor = KeyBERTTopicExtractor()
                print("[✅] Topic extractor initialized")
            except Exception as e:
                print(f"[WARNING] Topic extractor failed: {e}")
                topic_extractor = None

            try:
                classifier = TopicClassifier()
                print("[✅] Topic classifier initialized")
            except Exception as e:
                print(f"[WARNING] Topic classifier failed: {e}")
                classifier = None

            try:
                question_generator = QuestionGenerator()
                print("[✅] Question generator initialized")
            except Exception as e:
                print(f"[WARNING] Question generator failed: {e}")
                question_generator = None

            # Milvus connection with error handling
            milvus_connected = connect_milvus_safe()
            if milvus_connected:
                try:
                    sentence_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
                    dim = sentence_model.get_sentence_embedding_dimension()
                    milvus_collection = ensure_collection_safe("webpages", dim)
                    print("[✅] Sentence model and Milvus collection initialized")
                except Exception as e:
                    print(f"[WARNING] Sentence model/Milvus setup failed: {e}")
                    sentence_model = None
                    milvus_collection = None
            else:
                sentence_model = None
                milvus_collection = None

            # Initialize QuestionAnswerer ONCE and reuse
            try:
                if milvus_collection:
                    question_answerer = QuestionAnswerer(collection=milvus_collection)
                    print("[✅] QuestionAnswerer with Milvus initialized")
                else:
                    question_answerer = QuestionAnswerer()
                    print("[✅] QuestionAnswerer without Milvus initialized")
            except Exception as e:
                print(f"[WARNING] QuestionAnswerer initialization failed: {e}")
                question_answerer = None

            _models_initialized = True
            print("✅ Models initialization completed successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Critical model initialization failure: {e}")
            _models_initialized = False
            return False

def connect_milvus_safe():
    """Connect to Milvus with error handling"""
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"[✓] Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
        return True
    except Exception as e:
        print(f"[WARNING] Failed to connect to Milvus: {e}")
        print("[INFO] Server will continue without Milvus functionality")
        return False

def ensure_collection_safe(name: str, dim: int) -> Collection:
    """Ensure collection exists with proper schema - don't drop if already exists"""
    try:
        if utility.has_collection(name):
            print(f"[✅] Connected to existing Milvus collection '{name}'")
            return Collection(name=name)  # just load existing one 
        # Create fresh collection only if not exists
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
        ]
        schema = CollectionSchema(fields, description="Web page content embeddings")
        collection = Collection(name=name, schema=schema)
        index_params = {
            "metric_type": "COSINE",
            "index_type": "FLAT",
            "params": {}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        print(f"[✓] Created collection: {name} with COSINE index")
        return collection
    except Exception as e:
        print(f"[ERROR] Failed to create or connect collection: {e}")
        return None

# Initialize Flask app first
app = Flask(__name__)
qa_system = QuestionAnswerer(collection_name="webpages")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
app.config['SECRET_KEY'] = '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'

# MongoDB setup with error handling
try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    # Test connection
    client.admin.command('ping')
    db = client['chatbotDB']
    users_collection = db['users']
    scraped_urls_collection = db['scraped_urls']
    scraped_data_collection = db['scraped_data']
    otp_collection = db['otp_codes']
    conversation_collection = db['conversations']
    user_profiles_collection = db['user_profiles']
    user_sessions_collection = db['user_sessions']
    user_statistics_collection = db['user_statistics']
    login_logs_collection = db['login_logs']
    qa_collection = db['qa_pairs']
    print("[✓] MongoDB connected successfully")
except Exception as e:
    print(f"[ERROR] MongoDB connection failed: {e}")
    print("[FATAL] Server cannot start without MongoDB")
    exit(1)

# Add a basic health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'mongodb': 'connected',
            'milvus': 'connected' if milvus_collection else 'disabled',
            'models': 'loaded' if question_generator else 'loading'
        }
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'WebscrapQA Backend Server is running!',
        'version': '1.0.0',
        'endpoints': ['/health', '/signup', '/login', '/chat', '/process-chat']
    })

# server.py
def insert_to_milvus(collection: Collection, data_dict: dict):
    """Insert processed data into Milvus with error handling"""
    if not collection or not sentence_model:
        print("[WARNING] Milvus collection or sentence model not available")
        return None

    try:
        # Get collection schema to determine field count
        schema = collection.schema
        field_names = [field.name for field in schema.fields]
        field_count = len(field_names)
        
        ids, urls, titles, topics, contents, vectors, user_ids = [], [], [], [], [], [], []

        # make ids unique across multiple inserts
        base = collection.num_entities

        for i, (_page_key, info) in enumerate(data_dict.items(), start=1):
            content_text = info.get("content", "") or ""
            ids.append(base + i)
            urls.append(info.get("url", "") or "")
            titles.append(info.get("title", "") or "")
            topics.append(info.get("topic", "") or "")
            contents.append(content_text)
            vectors.append(
                sentence_model.encode(content_text, convert_to_numpy=True).tolist()
            )
            user_ids.append(str(info.get("user_id", "")))

        # Prepare rows based on actual schema field count
        if field_count == 6:
            # Old schema: id, url, title, topic, content, vector
            rows = [ids, urls, titles, topics, contents, vectors]
            print(f"[INFO] Using 6-field schema (without user_id)")
        elif field_count == 7:
            # New schema: id, url, title, topic, content, vector, user_id
            rows = [ids, urls, titles, topics, contents, vectors, user_ids]
            print(f"[INFO] Using 7-field schema (with user_id)")
        else:
            print(f"[ERROR] Unsupported schema with {field_count} fields: {field_names}")
            return None
        
        insert_result = collection.insert(rows)
        collection.flush()
        print(f"[✓] Inserted {len(ids)} records into Milvus")
        return insert_result

    except Exception as e:
        print(f"[ERROR] Milvus insertion failed: {e}")
        return None
    
def search_milvus(collection: Collection, query_text: str, top_k: int = 5):
    """Search similar content in Milvus with error handling"""
    if not collection or not sentence_model:
        return []
   
    try:
        # Load collection first
        collection.load() 
        # Generate query vector
        query_vector = sentence_model.encode(query_text, convert_to_numpy=True).tolist()
        # Search
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["url", "title", "content", "topic"]
        )
        return results[0] if results else []
    except Exception as e:
        print(f"[ERROR] Milvus search failed: {e}")
        return []

def send_email(recipient_email, subject, body):
    smtp_server = "smtp.gmail.com"
    smtp_port = 465
    sender_email = "kulmen0218@gmail.com"
    sender_password = "orctuebxvqzuqxeg"
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, message.as_string())
        server.quit()
        print(f"Email sent successfully to {recipient_email}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def process_page_data(item):
    """Process individual page data"""
    try:
        k, v = item
        v.pop("cleaned_content", None)
        
        if topic_extractor and classifier:
            v["topic"] = topic_extractor.extract_main_topic(v["lemmatized_content"])
            v["predicted_category"] = classifier.classify(v["lemmatized_content"])
        else:
            v["topic"] = "General"
            v["predicted_category"] = "General"      
        return k, v
    except Exception as e:
        print(f"[ERROR] Processing page data: {e}")
        return item

def process_qa_generation(item):
    """Generate QA pairs for individual page"""
    page_key, info = item
    try:
        topic = info.get("topic", "Unknown")
        category = info.get("predicted_category", "General")
        raw_content = info.get("content", "")
        title = info.get("title", "")
        url = info.get("url", "")
        # Extract contact details and enhance content
        contact_info = extract_contact_info(raw_content) 
        extra_info = ""
        if contact_info.get("mail"):
            extra_info += "\nEmail(s): " + ", ".join(contact_info["mail"])
        if contact_info.get("phone"):
            extra_info += "\nPhone Number(s): " + ", ".join(contact_info["phone"])
        if contact_info.get("address"):
            extra_info += "\nAddress: " + contact_info["address"]
        # Final content for QA generation
        content = f"{title}\n\n{raw_content.strip()}\n{extra_info.strip()}"
        # ✅ Generate QA pairs using the question generator
        qa_pairs = question_generator.generate_questions_and_answers(topic, category, content) 
        # ✅ Ensure qa_pairs is not empty
        if not qa_pairs:
            # Create a basic QA pair if generation fails
            qa_pairs = [{
                'question': f"What is this page about?",
                'short_answer': f"This page is about {topic}",
                'long_answer': f"This page discusses {topic} in the context of {category}. {content[:200]}..."
            }] 
        return page_key, {
            "url": url, 
            "title": title, 
            "topic": topic, 
            "qa_pairs": qa_pairs
        } 
    except Exception as e:
        print(f"❌ Error processing QA for {page_key}: {e}")
        # Return a fallback QA pair even on error
        return page_key, {
            "url": info.get("url", ""),
            "title": info.get("title", ""),
            "topic": info.get("topic", "Unknown"),
            "qa_pairs": [{
                'question': f"What can you tell me about this page?",
                'short_answer': "Content processing failed",
                'long_answer': "There was an issue processing this page content."
            }]
        }
    
def run_spider_subprocess(url, max_urls=20):
    try:
        # Get the absolute path to the crawler directory
        crawler_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crawler")
        run_spider_path = os.path.join(crawler_dir, "run_spider.py")
        print(f"[DEBUG] Running spider from: {crawler_dir}")
        print(f"[DEBUG] Spider script: {run_spider_path}")
        print(f"[DEBUG] Target URL: {url}")
        print(f"[DEBUG] Max URLs: {max_urls}")
        
        # Determine timeout based on URL type
        parsed_url = urlparse(url)
        timeout = 600 if 'wikipedia.org' in parsed_url.netloc else 300  # 10 mins for Wikipedia, 5 mins for others
        
        # Run subprocess with max_urls parameter
        result = subprocess.run(
            ["python", run_spider_path, url, str(max_urls)],  # Pass max_urls as argument
            capture_output=True,
            text=True,
            timeout=timeout,  # Dynamic timeout
            cwd=crawler_dir,  # Set working directory to crawler folder
            env={**os.environ, 'PYTHONPATH': crawler_dir}  # Add crawler dir to Python path
        )
        
        print(f"[DEBUG] Spider return code: {result.returncode}")
        
        if result.stderr:
            stderr_lines = result.stderr.strip().split('\n')
            for line in stderr_lines:
                if line.strip():
                    print(f"[SPIDER DEBUG] {line}")
                    
        if result.returncode != 0:
            print(f"[ERROR] Spider process failed with return code: {result.returncode}")
            return [url]  # Return at least the original URL
        
        # Debug the raw output
        print(f"[DEBUG] Raw spider stdout length: {len(result.stdout)}")
        print(f"[DEBUG] Raw spider stdout preview: {result.stdout[:200]}...")
        
        # Clean the stdout - remove any non-JSON lines
        output_lines = result.stdout.strip().split('\n')
        print(f"[DEBUG] Spider output has {len(output_lines)} lines")
        
        json_line = None
        # Find the line that looks like JSON (starts with [ and ends with ])
        for i, line in enumerate(output_lines):
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                json_line = line
                print(f"[DEBUG] Found JSON on line {i+1}: {line[:100]}...")
                break
                
        if not json_line:
            print(f"[WARNING] No JSON output found in spider result")
            print(f"[DEBUG] All output lines:")
            for i, line in enumerate(output_lines):
                print(f"[DEBUG] Line {i+1}: {line}")
            return [url]
            
        try:
            urls = json.loads(json_line)
            if isinstance(urls, list) and len(urls) > 0:
                print(f"[SUCCESS] Spider collected {len(urls)} URLs")
                # Debug first few URLs
                for i, spider_url in enumerate(urls[:5]):
                    print(f"[DEBUG] Spider URL {i+1}: {spider_url}")
                if len(urls) > 5:
                    print(f"[DEBUG] ... and {len(urls) - 5} more URLs")
                return urls  # Return all URLs, don't limit here
            else:
                print("[WARNING] Spider returned empty or invalid URL list")
                return [url]
        except json.JSONDecodeError as e:
            print(f"[ERROR] Spider returned invalid JSON: {e}")
            print(f"[DEBUG] Invalid JSON content: {json_line}")
            return [url]
            
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Spider process timed out after {timeout} seconds")
        return [url]
    except Exception as e:
        print(f"[ERROR] Failed to run spider: {e}")
        import traceback
        traceback.print_exc()
        return [url]
    
# Helper function to generate OTP for password reset
def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

@app.route('/test-spider', methods=['POST'])
def test_spider_endpoint():
    """Test spider functionality directly"""
    data = request.get_json()
    test_url = data.get('url', 'https://aapgs.com/')
    print(f"[TEST] Testing spider with URL: {test_url}")
    
    try:
        # Test the subprocess approach
        print("[TEST] Testing subprocess spider...")
        subprocess_urls = run_spider_subprocess(test_url)
        # Test the URLCollector approach  
        print("[TEST] Testing URLCollector...")
        from crawler.spider import URLCollector
        collector = URLCollector(test_url)
        collector_urls = collector.collect()
        return jsonify({
            'success': True,
            'subprocess_results': {
                'count': len(subprocess_urls),
                'urls': subprocess_urls[:10],  # First 10 for preview
                'sample_urls': subprocess_urls
            },
            'collector_results': {
                'count': len(collector_urls),
                'urls': collector_urls[:10],  # First 10 for preview  
                'sample_urls': collector_urls
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
    
def process_with_pipeline(url, content):
    global topic_extractor, classifier   # 👈 reuse global models
    try:
        cleaner = Cleaner()
        cleaned_content = cleaner.clean(content)
        if not cleaned_content:
            return None
        lemmatizer = TextLemmatizer()
        lemmatized_content = lemmatizer.lemmatize(cleaned_content)
        # ✅ Use already initialized global instances
        main_topic = topic_extractor.extract_main_topic(lemmatized_content) if topic_extractor else "General"
        predicted_category = classifier.classify(lemmatized_content) if classifier else "General"
        return {
            'cleaned_content': cleaned_content,
            'lemmatized_content': lemmatized_content,
            'topic': main_topic,
            'predicted_category': predicted_category
        }
    except Exception as e:
        print(f"[ERROR] Pipeline processing failed: {e}")
        return None

# Helper function to generate OTP for password reset
def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

# Helper functions for JWT tokens
def generate_token(user_id):
    expiration = datetime.now(timezone.utc) + timedelta(days=1)
    payload = {
        'exp': expiration,
        'iat': datetime.now(timezone.utc),
        'sub': str(user_id)
    }
    try:
        return pyjwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
    except Exception as e:
        print(f"Error generating token: {e}")
        return None

def verify_token(token):
    try:
        payload = pyjwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload['sub']
    except pyjwt.ExpiredSignatureError:
        print("Token has expired")
        return None
    except pyjwt.InvalidTokenError as e:
        print(f"Invalid token: {e}")
        return None

def save_conversation(user_id, username, user_message, bot_response):
    conversation_collection.insert_one({
        "user_id": ObjectId(user_id) if user_id else None,
        "username": username,
        "timestamp": datetime.utcnow(),
        "messages": [
            {"sender": "user", "text": user_message},
            {"sender": "bot", "text": bot_response}
        ]
    })

def is_greeting(message):
    greetings = [
        "hi", "hello", "hey", "how are you", "good morning", "good evening",
        "what's up", "howdy", "good afternoon"
    ]
    msg = message.lower().strip()
    return any(greet in msg for greet in greetings)

def is_question_related(query_embedding, selected_embeddings, threshold=0.4):
    if not selected_embeddings:
        return False
    similarities = cosine_similarity([query_embedding], selected_embeddings)
    max_similarity = np.max(similarities)
    return max_similarity >= threshold
   
# Middleware to check if user is authenticated
def token_required(f):
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'success': False, 'message': 'Token is missing'}), 401
        
        if token.startswith('Bearer '):
            token = token[7:]
            
        user_id = verify_token(token)
        if not user_id:
            return jsonify({'success': False, 'message': 'Token is invalid or expired'}), 401
        
        return f(user_id, *args, **kwargs)
    
    return decorated

# ------------------- SIGNUP -------------------
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    # Validate input data
    if not name or not email or not password:
        return jsonify({'success': False, 'message': 'Please provide all required fields'}), 400
    # Check if email already exists
    if users_collection.find_one({'email': email}):
        return jsonify({'success': False, 'message': 'Email already exists'}), 409
    # Hash the password for security
    hashed_password = generate_password_hash(password)
    # Insert new user into database
    user_id = users_collection.insert_one({
        'name': name, 
        'email': email, 
        'password': hashed_password,
        'created_at': datetime.now(),
        'last_login': None
    }).inserted_id
    
    return jsonify({
        'success': True, 
        'message': 'Signup successful',
        'userId': str(user_id),
        'username': name
    })

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({'success': False, 'message': 'Please provide email and password'}), 400
    user = users_collection.find_one({'email': email})
    if user and check_password_hash(user.get('password'), password):
        # Update last login timestamp
        users_collection.update_one(
            {'_id': user['_id']},
            {'$set': {'last_login': datetime.now()}}
        )
        # Track real login session
        login_logs_collection.insert_one({
            'user_id': user['_id'],
            'timestamp': datetime.now()
        })
        # Generate token
        token = generate_token(user['_id'])        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'username': user.get('name'),
            'userId': str(user['_id']),
            'email': user.get('email'),
            'token': token
        })
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

# ------------------- GOOGLE LOGIN -------------------
@app.route('/google-login', methods=['POST'])
def google_login():
    data = request.get_json()
    token = data.get('token')  
    if not token:
        return jsonify({'success': False, 'message': 'Token is required'}), 400  
    try:
        google_response = requests.get(
            f'https://www.googleapis.com/oauth2/v3/tokeninfo?id_token={token}'
        )
        if google_response.status_code != 200:
            return jsonify({'success': False, 'message': 'Invalid Google token'}), 401
        google_data = google_response.json()
        email = google_data.get('email')
        if not email:
            return jsonify({'success': False, 'message': 'Email not found in Google account'}), 400
        user = users_collection.find_one({'email': email})
        if not user:
            user_id = users_collection.insert_one({
                'name': google_data.get('name', email.split('@')[0]),
                'email': email,
                'password': None,
                'created_at': datetime.now(),
                'last_login': datetime.now(),
                'google_id': google_data.get('sub'),
                'picture': google_data.get('picture')
            }).inserted_id
        else:
            user_id = user['_id']
            users_collection.update_one(
                {'_id': user_id},
                {'$set': {
                    'last_login': datetime.now(),
                    'google_id': google_data.get('sub'),
                    'picture': google_data.get('picture', user.get('picture'))
                }}
            )
        # ✅ Track real login session for Google login
        login_logs_collection.insert_one({
            'user_id': user_id,
            'timestamp': datetime.now()
        })
        updated_user = users_collection.find_one({'_id': user_id})
        token = generate_token(user_id)
        return jsonify({
            'success': True,
            'message': 'Google login successful',
            'username': updated_user.get('name'),
            'userId': str(user_id),
            'email': email,
            'token': token
        })
    except Exception as e:
        print(f"Google login error: {e}")
        return jsonify({'success': False, 'message': 'Failed to authenticate with Google'}), 500

# ------------------- REQUEST PASSWORD RESET -------------------
@app.route('/request-password-reset', methods=['POST'])
def request_password_reset():
    data = request.get_json()
    email = data.get('email')
    if not email:
        return jsonify({'success': False, 'message': 'Email is required'}), 400
    # Check if user exists
    user = users_collection.find_one({'email': email})
    if not user:
        return jsonify({'success': True, 'message': 'If an account exists with this email, an OTP has been sent'})
    # Generate OTP
    otp = generate_otp()
    # Store OTP in database with expiration time (10 minutes)
    expiration_time = datetime.now() + timedelta(minutes=10)
    # Remove any existing OTPs for this email
    otp_collection.delete_many({'email': email})
    # Insert new OTP
    otp_collection.insert_one({
        'email': email,
        'otp': otp,
        'expires_at': expiration_time
    })
    # Prepare email content
    email_subject = "Your Password Reset OTP"
    email_body = f"""
    Hello {user.get('name', 'User')},
    
    You requested to reset your password for the WebscrapQA Chatbot.
    
    Your OTP code is: {otp}
    
    This code will expire in 10 minutes.
    
    If you did not request a password reset, please ignore this email.
    
    Best regards,
    WebscrapQA Team
    """
    # Send email with OTP
    email_sent = send_email(email, email_subject, email_body)
    if email_sent:
        print(f"OTP for {email}: {otp}")  # Still log to console for debugging
        return jsonify({
            'success': True,
            'message': 'If an account exists with this email, an OTP has been sent'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Failed to send OTP. Please try again later.'
        }), 500
    
# ------------------- VERIFY OTP -------------------
@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    email = data.get('email')
    otp = data.get('otp')  
    if not email or not otp:
        return jsonify({'success': False, 'message': 'Email and OTP are required'}), 400 
    # Find OTP in database
    otp_record = otp_collection.find_one({
        'email': email,
        'otp': otp,
        'expires_at': {'$gt': datetime.now()}  # Check if OTP is not expired
    }) 
    if not otp_record:
        return jsonify({'success': False, 'message': 'Invalid or expired OTP'}), 401
    return jsonify({
        'success': True,
        'message': 'OTP verified successfully'
    })

# ------------------- RESET PASSWORD -------------------
@app.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    email = data.get('email')
    otp = data.get('otp')
    new_password = data.get('newPassword')
    if not email or not otp or not new_password:
        return jsonify({'success': False, 'message': 'All fields are required'}), 400
    # Verify OTP
    otp_record = otp_collection.find_one({
        'email': email,
        'otp': otp,
        'expires_at': {'$gt': datetime.now()}
    })
    if not otp_record:
        return jsonify({'success': False, 'message': 'Invalid or expired OTP'}), 401
    # Hash new password
    hashed_password = generate_password_hash(new_password)
    # Update user's password
    result = users_collection.update_one(
        {'email': email},
        {'$set': {'password': hashed_password}}
    )
    if result.modified_count == 0:
        return jsonify({'success': False, 'message': 'Failed to update password'}), 500
    # Delete the used OTP
    otp_collection.delete_one({'_id': otp_record['_id']})
    return jsonify({
        'success': True,
        'message': 'Password reset successfully'
    })

# ------------------- SCRAPING ENDPOINT (UPDATED) -------------------
# ------------------- SCRAPING ENDPOINT (UPDATED) -------------------
@app.route('/chat', methods=['POST'])
def process_scrap():
    try:
        # ✅ Ensure models are ready before scraping
        if not ensure_models_ready():
            return jsonify({'success': False, 'message': 'Models not available'}), 500
        print("=" * 80)
        print("[DEBUG] 🚀 STARTING ENHANCED SCRAPING PROCESS")
        print("=" * 80)
        data = request.get_json()
        url = data.get('url')
        user_id = data.get('userId', '')
        user_object_id = ObjectId(user_id) if user_id else None
        use_crawler = data.get('useCrawler', False)
        if not url:
            return jsonify({'success': False, 'message': 'URL is required'}), 400
        socketio.emit('scraping_started', {'status': 'started'})
        print(f"[DEBUG] 📡 Scraping URL: {url}")
        print(f"[DEBUG] 👤 User ID: {user_id}")
        print(f"[DEBUG] 🕷️ Use Crawler: {use_crawler}")

        
        # ---------- STEP 1: URL COLLECTION (WITH DEBUGGING) ----------
        max_urls = 20  # Set your desired maximum
        links = [url]
        print(f"[DEBUG] Step 1a - Initial links: {len(links)}")

        if use_crawler:
            try:
                collector = URLCollector(url)
                collected_urls = collector.collect()
                if collected_urls:
                    print(f"[DEBUG] Step 1b - Collector found {len(collected_urls)} URLs")
                    links = collected_urls[:max_urls]
                    print(f"[DEBUG] Step 1c - After collector limit: {len(links)} URLs")
            except Exception as e:
                print(f"[WARNING] URL Collector failed: {e}")

        print(f"[DEBUG] Step 1d - Before spider check: {len(links)} URLs")

        if len(links) <= 2:
            try:
                print(f"[DEBUG] Step 1e - Running spider subprocess for max {max_urls} URLs...")
                spider_urls = run_spider_subprocess(url, max_urls)  # Pass max_urls parameter
                print(f"[DEBUG] Step 1f - Spider returned {len(spider_urls) if spider_urls else 0} URLs")
                if spider_urls:
                    print(f"[DEBUG] First 5 spider URLs: {[u[:60] + '...' if len(u) > 60 else u for u in spider_urls[:5]]}")
                    links = spider_urls  # Don't limit here, spider already limited
                    print(f"[DEBUG] Step 1g - After spider assignment: {len(links)} URLs")
            except Exception as e:
                print(f"[WARNING] Spider failed: {e}")

        print(f"[DEBUG] Step 1h - Before deduplication: {len(links)} URLs")

        # Remove duplicates while preserving order
        original_count = len(links)
        links = list(dict.fromkeys(links))
        print(f"[DEBUG] Step 1i - After deduplication: {len(links)} URLs (removed {original_count - len(links)} duplicates)")

        # Apply final safety limit
        links = links[:max_urls]
        print(f"[DEBUG] Step 1j - Final URL list: {len(links)} URLs")

        print(f"[INFO] Final URL list: {len(links)} unique URLs to scrape")

        # ---------- STEP 2: SCRAPING (UPDATED) ----------
        chromedriver_path = r"D:\chromedriver-win64 (2)\chromedriver-win64\chromedriver.exe"
        scraper = SeleniumScraper(driver_path=chromedriver_path)  
        scraped_pages = scraper.scrape(links)

        print(f"[DEBUG] Scraper returned {len(scraped_pages)} pages")

        if not scraped_pages:
            socketio.emit('scraping_failed', {'status': 'error', 'message': 'No content extracted'})
            return jsonify({'success': False, 'message': 'Scraping failed - no content extracted'}), 500

        # ---------- STEP 3: CLEANING & LEMMATIZATION (FIXED) ----------
        cleaner = Cleaner()
        lemmatizer = TextLemmatizer()
        processed_data = {}

        print(f"[DEBUG] Processing {len(scraped_pages)} scraped pages...")

        for page_key, page_info in scraped_pages.items():
            raw_content = page_info.get('content', '')
            if len(raw_content.strip()) < 50:
                print(f"[DEBUG] Skipping {page_key} - content too short ({len(raw_content)} chars)")
                continue
                
            contact_info = extract_contact_info(raw_content)
            enhanced_content = raw_content
            
            # Enhance content with contact info
            if contact_info.get("mail"):
                for email in contact_info["mail"]:
                    enhanced_content = enhanced_content.replace(email, f"mail {email}")
            if contact_info.get("phone"):
                for phone in contact_info["phone"]:
                    enhanced_content = enhanced_content.replace(phone, f"phone {phone}")
            if contact_info.get("address"):
                addr = contact_info["address"]
                enhanced_content = enhanced_content.replace(addr, f"address {addr}")
            
            try:
                cleaned = cleaner.clean(enhanced_content.strip())
                lemmatized = lemmatizer.lemmatize(cleaned)
                
                processed_data[page_key] = {
                    **page_info,
                    "content": enhanced_content.strip(),
                    "cleaned_content": cleaned,
                    "lemmatized_content": lemmatized
                }
                print(f"[DEBUG] Processed {page_key} - content length: {len(enhanced_content)}")
            except Exception as e:
                print(f"[ERROR] Failed to process {page_key}: {e}")
                continue

        print(f"[DEBUG] Successfully processed {len(processed_data)} pages for topic classification")

        # ---------- STEP 4: TOPIC CLASSIFICATION ----------
        print("[INFO] 🚀 Starting threaded topic classification...")
        classified_data = batch_process_with_threading(processed_data, max_workers=1)
        socketio.emit('scraping_progress', {'progress': 65})

        # ---------- STEP 5: SAVE TO MONGODB & MILVUS ----------
        responses = []
        saved_count = 0
        milvus_inserted = 0
        for page_key, page_info in classified_data.items():
            current_url = page_info.get('url', '')
            title = page_info.get('title', '') or ''
            content = page_info.get('content', '') or ''
            topic = page_info.get('topic', 'General')
            category = page_info.get('predicted_category', 'General')
            content_hash = page_info.get('content_hash', f"hash_{page_key}")
            scraped_data_doc = {
                'url': current_url,
                'title': title,
                'content': content,
                'content_hash': content_hash,
                'cleaned_content': page_info.get('cleaned_content', ''),
                'lemmatized_content': page_info.get('lemmatized_content', ''),
                'topic': topic,
                'predicted_category': category,
                'timestamp': datetime.now(),
                'user_id': user_object_id,
                'access_count': 1,
                'created_at': datetime.now()
            }
            scraped_url_doc = {
                'user_id': user_object_id,
                'url': current_url,
                'title': title,
                'topic': topic,
                'predicted_category': category,
                'timestamp': datetime.now(),
                'is_root': (current_url == url),
                'content_hash': content_hash,
                'content_preview': content[:200] + '...' if len(content) > 200 else content,
                'scraped_at': datetime.now(),
                'created_at': datetime.now()
            }
            scraped_data_collection.update_one(
                {'url': current_url, 'user_id': user_object_id},
                {'$set': scraped_data_doc},
                upsert=True
            )
            scraped_urls_collection.update_one(
                {'user_id': user_object_id, 'url': current_url},
                {'$set': scraped_url_doc},
                upsert=True
            )
            saved_count += 1
            if milvus_collection and sentence_model:
                single_data = {page_key: {
                    'url': current_url,
                    'title': title,
                    'topic': topic,
                    'content': content
                }}
                if insert_to_milvus(milvus_collection, single_data):
                    milvus_inserted += 1
            responses.append({
                'url': current_url,
                'title': title,
                'topic': topic,
                'category': category,
                'content_length': len(content)
            })
        socketio.emit('scraping_progress', {'progress': 80})

        # ---------- STEP 6: BACKGROUND QA GENERATION ----------
        def background_qa():
            try:
                print("[BACKGROUND] 🛠️ Generating QA pairs in background...")
                total_qa_stored = 0
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(process_qa_generation_threaded, item)
                               for item in classified_data.items()]
                    for future in as_completed(futures):
                        try:
                            key, result = future.result()
                            if result and result.get('qa_pairs'):
                                qa_doc = {
                                    'url': result.get('url', ''),
                                    'title': result.get('title', ''),
                                    'topic': result.get('topic', 'General'),
                                    'qa_pairs': result.get('qa_pairs', []),
                                    'user_id': user_object_id,
                                    'timestamp': datetime.now(),
                                    'total_pairs': len(result.get('qa_pairs', [])),
                                    'created_at': datetime.now()
                                }
                                qa_collection.update_one(
                                    {'url': result.get('url', ''), 'user_id': user_object_id},
                                    {'$set': qa_doc},
                                    upsert=True
                                )
                                total_qa_stored += len(result.get('qa_pairs', []))
                        except Exception as e:
                            print(f"[ERROR] QA generation failed: {e}")
                print(f"[BACKGROUND] ✅ QA generation completed: {total_qa_stored} pairs stored.")
            except Exception as e:
                print(f"[BACKGROUND ERROR] {e}")
        Thread(target=background_qa, daemon=True).start()
        socketio.emit('scraping_progress', {'progress': 100})
        socketio.emit('scraping_completed', {
            'status': 'completed',
            'total_pages': len(responses),
            'milvus_vectors_inserted': milvus_inserted
        })
        return jsonify({
            'success': True,
            'message': f'Successfully processed {len(responses)} unique pages. QA will be generated in background.',
            'results': responses,
            'milvus_vectors': milvus_inserted
        })
    except Exception as e:
        print(f"[FATAL ERROR] /chat: {e}")
        socketio.emit('scraping_failed', {'status': 'error', 'message': str(e)})
        return jsonify({'success': False, 'message': str(e)}), 500

def refresh_milvus_from_mongo():
    """Rebuild Milvus collection from latest MongoDB scraped_data content"""
    try:
        from sentence_transformers import SentenceTransformer
        from question.embedding import connect_milvus, ensure_collection
        from pymilvus import Collection

        # Mongo connection
        documents = list(scraped_data_collection.find({}))
        if not documents:
            print("[WARNING] No scraped data found in MongoDB")
            return

        # Model & collection setup
        model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        dim = model.get_sentence_embedding_dimension()
        connect_milvus("localhost", 19530)
        collection = ensure_collection("webpages", dim)

        # Prepare rows
        ids, urls, titles, topics, contents, vectors, user_ids = [], [], [], [], [], [], []
        for idx, doc in enumerate(documents, start=1):
            ids.append(idx)
            urls.append(doc.get("url", ""))
            titles.append(doc.get("title", ""))
            topics.append(doc.get("topic", ""))
            contents.append(doc.get("content", ""))
            vectors.append(model.encode(doc.get("content", "") or "", convert_to_numpy=True).tolist())
            user_ids.append(str(doc.get("user_id", "")))
        rows = [ids, urls, titles, topics, contents, vectors, user_ids]
        collection.insert(rows)
        collection.flush()
        index_params = {"metric_type": "COSINE", "index_type": "FLAT", "params": {}}
        collection.create_index(field_name="vector", index_params=index_params)
        print(f"[✓] Milvus refreshed with {len(ids)} documents")
    except Exception as e:
        print(f"[ERROR] Failed to refresh Milvus: {e}")

# ------------------- GET USER'S RECENT URLS -------------------
@app.route('/recent-urls', methods=['GET'])
def get_recent_urls():
    user_id = request.args.get('userId')
    username = request.args.get('username')
    user_object_id = None
    if user_id:
        try:
            user_object_id = ObjectId(user_id)
        except Exception as e:
            return jsonify({'success': False, 'message': 'Invalid userId'}), 400

    # Build query to filter by user
    query = {}
    if user_object_id:
        query['user_id'] = user_object_id
    elif username:
        # If using username, get user_id first
        user = users_collection.find_one({'name': username})
        if user:
            query['user_id'] = user['_id']
        else:
            return jsonify({'success': False, 'message': 'User not found'}), 404
    else:
        return jsonify({'success': False, 'message': 'Either userId or username is required'}), 400

    # Only get root URLs for this specific user
    query['is_root'] = True

    try:
        recent_urls = list(scraped_urls_collection.find(
            query,
            {
                '_id': 0,
                'url': 1,
                'title': 1,
                'timestamp': 1,
                'content_preview': 1,
                'content_hash': 1,
                'topic': 1,
                'predicted_category': 1
            }
        ).sort('timestamp', -1).limit(10))

        # Format timestamps
        for url_data in recent_urls:
            if 'timestamp' in url_data:
                url_data['timestamp'] = url_data['timestamp'].isoformat()
        print(f"[DEBUG] Found {len(recent_urls)} recent URLs for user {user_id}")
        return jsonify({
            'success': True,
            'urls': recent_urls,
            'count': len(recent_urls)
        })
    except Exception as e:
        print(f"[ERROR] Failed to get recent URLs: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/generate-qa', methods=['POST'])
def generate_qa_pairs():
    try:
        # Generate QA pairs from all processed data
        qg = QuestionGenerator()
        qg.save_all_to_file()
        return jsonify({
            'success': True,
            'message': 'QA pairs generated successfully'
        })
    except Exception as e:
        print(f"[ERROR] QA generation failed: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to generate QA pairs: {str(e)}'
        }), 500
    
# -------------------- CHAT PROCESSING ENDPOINT -------------------
@app.route('/process-chat', methods=['POST'])
def process_chat():
    data = request.get_json()
    message = (data.get('message') or '').strip()
    user_id = data.get('userId', '')
    username = data.get('username', '')
    selected_urls = data.get('selectedUrls', [])
    conversation_id = data.get('conversationId')

    print(f"[DEBUG] Processing chat - Message: {message[:50]}...")
    print(f"[DEBUG] Selected URLs: {selected_urls}")

    handler = get_response_handler()
    user_object_id = ObjectId(user_id) if user_id else None
    bot_response = None

    # --- Handle greetings & basic identity queries ---
    if not message or message.lower() in ["hi", "hello", "hey"]:
        bot_response = "👋 Hello! How can I assist you today?"
    elif message.lower() in ["what is my name", "who am i", "tell me my name"]:
        bot_response = f"🤖 Your current name is {username}."
    else:
        # --- Small talk / idle handlers ---
        for check in [
            handler._is_greeting,
            handler._is_gratitude,
            handler._is_confused,
            handler._handle_self_reference,
            handler._handle_idle_talk
        ]:
            try:
                resp = check(message.lower())
                if resp:
                    bot_response = resp
                    break
            except Exception as e:
                print(f"[WARNING] Handler check failed: {e}")

        # --- Main QA logic ---
        if not bot_response:
            try:
                root_url = selected_urls[0] if selected_urls else None

                # ✅ Reuse globally initialized QA system
                result = qa_system.answer_question(
                    question=message,
                    root_url=root_url,
                    user_id=user_object_id,
                    selected_urls=selected_urls
                )

                if result.get('success') and result.get('answers'):
                    best_answer = result['answers'][0]
                    bot_response = best_answer.get('long_answer', '')

                    # add title or url for context
                    if best_answer.get('title'):
                        bot_response += f"\n\n📚 Source: {best_answer['title']}"
                    elif best_answer.get('url'):
                        bot_response += f"\n\n📚 Source: {best_answer['url']}"
                else:
                    bot_response = "I couldn't find a specific answer in the scraped content."
            except Exception as e:
                print(f"[ERROR] QA processing failed: {e}")
                bot_response = "⚠️ An error occurred while processing your question."

    # --- Save conversation ---
    new_message_pair = [
        {"sender": "user", "text": message},
        {"sender": "bot", "text": bot_response}
    ]
    try:
        if conversation_id:
            conversation_object_id = ObjectId(conversation_id)
            update_result = conversation_collection.update_one(
                {"_id": conversation_object_id},
                {"$push": {"messages": {"$each": new_message_pair}},
                 "$set": {"timestamp": datetime.now(timezone.utc)}}
            )
            if update_result.modified_count == 0:
                raise Exception("Conversation not found")
        else:
            result = conversation_collection.insert_one({
                "user_id": user_object_id,
                "username": username,
                "timestamp": datetime.now(timezone.utc),
                "messages": new_message_pair
            })
            conversation_id = str(result.inserted_id)

    except Exception as e:
        print(f"[WARNING] Conversation update failed: {e}")
        result = conversation_collection.insert_one({
            "user_id": user_object_id,
            "username": username,
            "timestamp": datetime.now(timezone.utc),
            "messages": new_message_pair
        })
        conversation_id = str(result.inserted_id)

    return jsonify({
        'success': True,
        'message': message,
        'response': bot_response,
        'conversationId': conversation_id,
        'semantic_results': [],
        'selected_urls': selected_urls
    })

# =================== 3. ADD DEBUGGING ENDPOINT ===================
@app.route('/debug-db', methods=['GET'])
def debug_database():
    """Debug endpoint to check what's in MongoDB"""
    user_id = request.args.get('userId')
    try:
        user_object_id = ObjectId(user_id) if user_id else None
        # Count documents in each collection
        stats = {
            'scraped_data_total': scraped_data_collection.count_documents({}),
            'scraped_urls_total': scraped_urls_collection.count_documents({}),
            'qa_pairs_total': qa_collection.count_documents({}),
        }
        if user_object_id:
            stats.update({
                'scraped_data_user': scraped_data_collection.count_documents({'user_id': user_object_id}),
                'scraped_urls_user': scraped_urls_collection.count_documents({'user_id': user_object_id}),
                'qa_pairs_user': qa_collection.count_documents({'user_id': user_object_id}),
            })
            # Get sample data
            sample_scraped = list(scraped_data_collection.find({'user_id': user_object_id}).limit(2))
            sample_qa = list(qa_collection.find({'user_id': user_object_id}).limit(2))  
            stats['sample_scraped'] = [{'url': doc.get('url'), 'title': doc.get('title')} for doc in sample_scraped]
            stats['sample_qa'] = [{'url': doc.get('url'), 'qa_count': len(doc.get('qa_pairs', []))} for doc in sample_qa]
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# ------------------- SEMANTIC SEARCH ENDPOINT -------------------
@app.route('/semantic-search', methods=['POST'])
def semantic_search():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({'success': False, 'message': 'Query is required'}), 400
    
    try:
        if not milvus_collection:
            return jsonify({'success': False, 'message': 'Milvus collection not available'}), 500
            
        # Search using Milvus
        search_results = search_milvus(milvus_collection, query, top_k=5)
        results = []
        for hit in search_results:
            results.append({
                'url': hit.entity.get('url', ''),
                'title': hit.entity.get('title', 'No title'),
                'content_preview': hit.entity.get('content', '')[:200] + '...' if len(hit.entity.get('content', '')) > 200 else hit.entity.get('content', ''),
                'similarity_score': 1.0 - hit.distance,  # Convert distance to similarity
                'topic': hit.entity.get('topic', '')
            })
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error performing semantic search: {str(e)}'
        }), 500

# Add this function to your server.py
def get_related_urls_for_domain(user_object_id, root_url, scraped_urls_collection):
    """
    Get all URLs related to a root URL, handling different domain structures
    """
    from urllib.parse import urlparse 
    matched_urls = set()
    matched_urls.add(root_url)  # Add root URL itself
    parsed_root = urlparse(root_url)
    root_domain = parsed_root.netloc
    if "wikipedia.org" in root_domain.lower():
        # For Wikipedia: find all URLs from same domain that were crawled in the same session
        print(f"[DEBUG] Wikipedia domain detected: {root_domain}")
        # Get the timestamp of the root URL
        root_doc = scraped_urls_collection.find_one({
            'user_id': user_object_id,
            'url': root_url,
            'is_root': True
        })
        if root_doc and root_doc.get('timestamp'):
            crawl_time = root_doc['timestamp']
            # Find all URLs crawled in the same session (same day)
            start_time = crawl_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = crawl_time.replace(hour=23, minute=59, second=59, microsecond=999999) 
            cursor = scraped_urls_collection.find({
                'user_id': user_object_id,
                'timestamp': {'$gte': start_time, '$lte': end_time},
                'url': {'$regex': f'https?://{re.escape(root_domain)}'}  # Same domain
            })
            for doc in cursor:
                matched_urls.add(doc['url'])
            print(f"[DEBUG] Wikipedia: Found {len(matched_urls)} related URLs")
    else:
        # For regular websites: use hierarchical matching (current logic)
        print(f"[DEBUG] Regular domain detected: {root_domain}")
        cursor = scraped_urls_collection.find({
            'user_id': user_object_id,
            'url': {'$regex': f'^{re.escape(root_url)}'}  # URLs that start with root_url
        })
        for doc in cursor:
            matched_urls.add(doc['url'])  
        print(f"[DEBUG] Regular: Found {len(matched_urls)} related URLs")
    return list(matched_urls)

# Update your /generate-faq endpoint
@app.route('/generate-faq', methods=['POST'])
def generate_faq():
    data = request.get_json()
    selected_urls = data.get('selectedUrls', [])
    user_id = data.get('userId', '')
    if not selected_urls:
        return jsonify({'success': False, 'message': 'No URLs provided'}), 400
    try:
        user_object_id = ObjectId(user_id) if user_id else None
        matched_urls = set()
        # Use the new function for each selected URL
        for root_url in selected_urls:
            related_urls = get_related_urls_for_domain(user_object_id, root_url, scraped_urls_collection)
            matched_urls.update(related_urls)
        print(f"[DEBUG] Selected URLs: {selected_urls}")
        print(f"[DEBUG] Total matched URLs (including children): {len(matched_urls)}")

        if not matched_urls:
            return jsonify({
                'success': True, 
                'faq': [],
                'debug_info': {
                    'selected_urls': selected_urls,
                    'matched_urls': 0,
                    'message': 'No matching URLs found'
                }
            })

        # Rest of the function remains the same...
        faq_entries = []
        cursor = qa_collection.find({
            'url': {'$in': list(matched_urls)},
            'user_id': user_object_id
        })

        for doc in cursor:
            qa_list = doc.get('qa_pairs', [])
            topic = doc.get('topic', 'General')
            doc_title = doc.get('title', 'Untitled')
            doc_url = doc.get('url', '')
            for qa in qa_list:
                faq_entries.append({
                    'question': qa.get('question', ''),
                    'answer': qa.get('long_answer') or qa.get('short_answer', ''),
                    'source': doc_title,
                    'topic': topic,
                    'source_url': doc_url
                })
        topics_found = set()
        for faq in faq_entries:
            topics_found.add(faq.get('topic', 'General'))
        return jsonify({
            'success': True, 
            'faq': faq_entries,
            'debug_info': {
                'selected_urls': selected_urls,
                'matched_urls': len(matched_urls),
                'total_faq_entries': len(faq_entries),
                'topics_found': list(topics_found),
                'sample_matched_urls': list(matched_urls)[:10]
            }
        })
    except Exception as e:
        print(f"[ERROR] FAQ generation failed: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'message': f'Error generating FAQ: {str(e)}'
        }), 500

def generate_questions_from_text(text):
    """
    Dummy question generator — replace with NLP logic as needed.
    """
    import re
    sentences = re.split(r'[.?!]\s+', text)
    questions = [s.strip() + '?' for s in sentences if len(s.split()) > 5 and 'you' not in s.lower()]
    return questions[:3]  # Return top 3

@app.route('/delete-url', methods=['POST'])
def delete_url():
    data = request.json
    user_id = data.get('userId')
    root_url = data.get('url')
    if not user_id or not root_url:
        return jsonify({'success': False, 'error': 'Missing userId or URL'}), 400
    try:
        user_object_id = ObjectId(user_id) 
        # 1. Find the root document for THIS USER ONLY
        root_doc = scraped_urls_collection.find_one({
            'user_id': user_object_id,
            'url': root_url,
            'is_root': True
        })
        if not root_doc:
            return jsonify({'success': False, 'error': 'Root URL not found for this user'}), 404
        crawl_time = root_doc.get('timestamp')
        if not crawl_time:
            return jsonify({'success': False, 'error': 'Missing timestamp on root URL'}), 500

        # 2. Identify crawl group (same user, same crawl session)
        start_time = crawl_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = crawl_time.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Get all URLs from the same crawl session for THIS USER ONLY
        crawl_group = list(scraped_urls_collection.find({
            'user_id': user_object_id,  # ✅ CRITICAL: Filter by user_id
            'timestamp': {'$gte': start_time, '$lte': end_time}
        }))
        urls_to_delete = [doc['url'] for doc in crawl_group if 'url' in doc]

        # 3. Delete from all collections - but ONLY for this user
        result1 = scraped_urls_collection.delete_many({
            'url': {'$in': urls_to_delete},
            'user_id': user_object_id  # ✅ CRITICAL: Only delete user's data
        })
        result2 = scraped_data_collection.delete_many({
            'url': {'$in': urls_to_delete},
            'user_id': user_object_id  # ✅ CRITICAL: Only delete user's data
        })
        result3 = qa_collection.delete_many({
            'url': {'$in': urls_to_delete},
            'user_id': user_object_id  # ✅ CRITICAL: Only delete user's data
        })
        print(f"[DELETE] User {user_id} deleted {len(urls_to_delete)} URLs")
        print(f"[DELETE] Deleted from scraped_urls: {result1.deleted_count}")
        print(f"[DELETE] Deleted from scraped_data: {result2.deleted_count}")
        print(f"[DELETE] Deleted from qa_pairs: {result3.deleted_count}")
        return jsonify({
            'success': True, 
            'deleted_count': len(urls_to_delete),
            'message': f'Successfully deleted {len(urls_to_delete)} URLs from your account'
        })
    except Exception as e:
        print(f"[ERROR] /delete-url failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
# Helper function for generating session summaries
def generate_session_summary(messages):
    """Generate a summary from the first user message"""
    if not messages:
        return "Empty conversation"
    # Find first user message
    for msg in messages:
        if msg.get("sender") == "user":
            text = msg.get("text", "")
            if len(text) > 50:
                return text[:50] + "..."
            return text
    
    return "No user messages"

# ------------------- GET CHAT SESSIONS -------------------
@app.route('/chat-sessions', methods=['GET'])
def get_chat_sessions():
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({"success": False, "message": "Missing userId"}), 400
    try:
        user_object_id = ObjectId(user_id)
        # ✅ CRITICAL: Only get conversations for this specific user
        sessions = list(conversation_collection.find(
            {"user_id": user_object_id},  # ✅ Filter by user_id
            {
                "_id": 1,
                "timestamp": 1,
                "messages": 1,
                "username": 1,
                "user_id": 1
            }
        ).sort("timestamp", -1).limit(20))
        print(f"[DEBUG] Found {len(sessions)} chat sessions for user {user_id}")
        formatted_sessions = []
        for session in sessions:
            formatted_sessions.append({
                "_id": str(session["_id"]),
                "date": session.get("timestamp"),
                "timestamp": session.get("timestamp"),
                "messages": session.get("messages", []),
                "username": session.get("username", "Unknown"),
                "summary": generate_session_summary(session.get("messages", []))
            })
        return jsonify({
            "success": True, 
            "sessions": formatted_sessions,
            "count": len(formatted_sessions)
        })
    except Exception as e:
        print(f"[ERROR] Error fetching chat sessions: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# ------------------- GET INDIVIDUAL CHAT SESSION -------------------
@app.route('/chat-session/<conversation_id>', methods=['GET'])
def get_chat_session(conversation_id):
    try:
        object_id = ObjectId(conversation_id)
        conversation = conversation_collection.find_one({"_id": object_id})
        if not conversation:
            return jsonify({"success": False, "message": "Conversation not found"}), 404
        # Format response
        formatted = {
            "_id": str(conversation["_id"]),
            "messages": conversation.get("messages", []),
            "username": conversation.get("username", "Unknown"),
            "timestamp": conversation.get("timestamp"),
        }
        return jsonify({"success": True, "session": formatted})
    except Exception as e:
        print(f"[ERROR] Failed to fetch chat session {conversation_id}: {e}")
        return jsonify({"success": False, "message": f"Invalid session ID: {str(e)}"}), 500

# ------------------- SAVE CONVERSATION -------------------
@app.route('/save-conversation', methods=['POST'])
def save_conversation_endpoint():
    data = request.json
    user_id = data.get('userId')
    username = data.get('username')
    messages = data.get('messages')
    if not user_id or not messages:
        return jsonify({"success": False, "message": "Missing data"}), 400

    # Prevent saving if only the welcome message exists
    if len(messages) == 1 and messages[0].get('sender') == 'bot' and "analyze content" in messages[0].get('text', '').lower():
        return jsonify({"success": False, "message": "Empty chat. Not saved."})
    try:
        user_object_id = ObjectId(user_id)
        conversation = {
            "user_id": user_object_id,
            "username": username,
            "messages": messages,
            "timestamp": datetime.utcnow()
        }
        result = conversation_collection.insert_one(conversation)
        print(f"[DEBUG] Saved new conversation with ID: {result.inserted_id}")
        return jsonify({
            "success": True,
            "message": "Conversation saved",
            "conversationId": str(result.inserted_id)
        })
    except Exception as e:
        print(f"[ERROR] Error saving conversation: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# ------------------- UPDATE CONVERSATION -------------------
@app.route('/update-conversation', methods=['PUT'])
def update_conversation():
    data = request.json
    conversation_id = data.get('conversationId')
    messages = data.get('messages')
    if not conversation_id or not messages:
        return jsonify({"success": False, "message": "Missing data"}), 400
    try:
        object_id = ObjectId(conversation_id)
        result = conversation_collection.update_one(
            {"_id": object_id},
            {
                "$set": {
                    "messages": messages,
                    "timestamp": datetime.utcnow()
                }
            }
        )
        if result.modified_count > 0:
            print(f"[DEBUG] Updated conversation {conversation_id}")
            return jsonify({"success": True, "message": "Conversation updated"})
        else:
            return jsonify({"success": False, "message": "Conversation not found or unchanged"}), 404
    except Exception as e:
        print(f"[ERROR] Error updating conversation: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# ------------------- DELETE CHAT SESSION -------------------
@app.route('/delete-chat-session', methods=['POST'])
def delete_chat_session():
    data = request.json
    user_id = data.get('userId')
    conversation_id = data.get('conversationId')
    if not user_id or not conversation_id:
        return jsonify({"success": False, "message": "Missing userId or conversationId"}), 400
    try:
        user_object_id = ObjectId(user_id)
        conversation_object_id = ObjectId(conversation_id)

        # ✅ CRITICAL: Only delete conversations that belong to this user
        result = conversation_collection.delete_one({
            "_id": conversation_object_id,
            "user_id": user_object_id  # ✅ Ensure user owns this conversation
        })

        if result.deleted_count == 1:
            print(f"[DELETE] User {user_id} deleted conversation {conversation_id}")
            return jsonify({"success": True, "message": "Conversation deleted successfully"})
        else:
            return jsonify({"success": False, "message": "Conversation not found or unauthorized"}), 404

    except Exception as e:
        print(f"[ERROR] Failed to delete conversation {conversation_id}: {e}")
        return jsonify({"success": False, "message": str(e)}), 500
  
@app.route('/user-profile', methods=['GET'])
def get_user_profile():
    user_id = request.args.get('userId')

    if not user_id:
        return jsonify({'success': False, 'message': 'User ID is required'}), 400

    try:
        user_object_id = ObjectId(user_id)

        # Get user basic info
        user = users_collection.find_one({'_id': user_object_id})
        if not user:
            return jsonify({'success': False, 'message': 'User not found'}), 404

        # Extended info
        profile = user_profiles_collection.find_one({'user_id': user_object_id})
        stats = user_statistics_collection.find_one({'user_id': user_object_id})
        recent_session = user_sessions_collection.find_one(
            {'user_id': user_object_id},
            sort=[('timestamp', -1)]
        )

        # ✅ New stats calculated live
        total_conversations = conversation_collection.count_documents({'user_id': user_object_id})
        urls_scraped = scraped_urls_collection.count_documents({'user_id': user_object_id, 'is_root': True})
        
        total_messages = 0
        conversations = conversation_collection.find({'user_id': user_object_id})
        for conv in conversations:
            total_messages += len(conv.get('messages', []))

        # ✅ Accurate login session count from login_logs
        total_login_sessions = login_logs_collection.count_documents({'user_id': user_object_id})

        # Final profile payload
        user_profile = {
            'userId': str(user['_id']),
            'username': user.get('name', ''),
            'email': user.get('email', ''),
            'firstName': profile.get('first_name', '') if profile else '',
            'lastName': profile.get('last_name', '') if profile else '',
            'bio': profile.get('bio', '') if profile else '',
            'location': profile.get('location', '') if profile else '',
            'timezone': profile.get('timezone', 'UTC') if profile else 'UTC',
            'language': profile.get('language', 'English') if profile else 'English',
            'joinDate': user.get('created_at', datetime.now()).strftime('%B %d, %Y'),
            'lastSeen': user.get('last_login', datetime.now()).strftime('%B %d, %Y at %I:%M %p'),
            'isOnline': recent_session is not None and recent_session.get('is_active', False),
            'accountType': profile.get('account_type', 'Free User') if profile else 'Free User',
            'isVerified': profile.get('is_verified', True) if profile else True,
            'avatar': profile.get('avatar', '') if profile else '',
            'memberSince': user.get('created_at', datetime.now()).strftime('%B %d, %Y'),

            # 📊 Statistics
            'totalMessages': total_messages,
            'urlsScraped': urls_scraped,
            'totalConversations': total_conversations,
            'totalSessions': total_login_sessions,  # ✅ real login sessions
            'successRate': stats.get('success_rate', 86) if stats else 86,
            'currentSessionTime': '0h 0m',
            'todayTime': stats.get('today_time', '2h 15m') if stats else '2h 15m'
        }

        return jsonify({'success': True, 'profile': user_profile})

    except Exception as e:
        print(f"[ERROR] Failed to get user profile: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ------------------- UPDATE USER PROFILE -------------------
@app.route('/update-profile', methods=['PUT'])
def update_user_profile():
    data = request.get_json()
    user_id = data.get('userId')
    
    if not user_id:
        return jsonify({'success': False, 'message': 'User ID is required'}), 400
    
    try:
        user_object_id = ObjectId(user_id)
        
        # Update user_profiles collection
        profile_data = {
            'user_id': user_object_id,
            'first_name': data.get('firstName', ''),
            'last_name': data.get('lastName', ''),
            'bio': data.get('bio', ''),
            'location': data.get('location', ''),
            'timezone': data.get('timezone', 'UTC'),
            'language': data.get('language', 'English'),
            'updated_at': datetime.now()
        }
        
        # Upsert profile data
        user_profiles_collection.update_one(
            {'user_id': user_object_id},
            {'$set': profile_data},
            upsert=True
        )
        
        # Update basic user info if name changed
        if data.get('firstName') or data.get('lastName'):
            full_name = f"{data.get('firstName', '')} {data.get('lastName', '')}".strip()
            if full_name:
                users_collection.update_one(
                    {'_id': user_object_id},
                    {'$set': {'name': full_name}}
                )
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully'
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to update profile: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ------------------- GET USER STATISTICS -------------------
@app.route('/user-stats', methods=['GET'])
def get_user_statistics():
    user_id = request.args.get('userId')
    
    if not user_id:
        return jsonify({'success': False, 'message': 'User ID is required'}), 400

    try:
        user_object_id = ObjectId(user_id)

        # 1. ✅ Count total conversations
        total_conversations = conversation_collection.count_documents({'user_id': user_object_id})

        # 2. ✅ Count total scraped root URLs
        total_scraped_urls = scraped_urls_collection.count_documents({
            'user_id': user_object_id,
            'is_root': True
        })

        # 3. ✅ Count total messages in all conversations
        total_messages = 0
        all_convs = conversation_collection.find({'user_id': user_object_id})
        for conv in all_convs:
            total_messages += len(conv.get('messages', []))

        # 4. ✅ Count total login sessions
        total_sessions = login_logs_collection.count_documents({'user_id': user_object_id})

        # 5. ✅ Count total user questions
        total_questions = 0
        all_convs.rewind()
        for conv in all_convs:
            messages = conv.get('messages', [])
            total_questions += len([msg for msg in messages if msg.get('sender') == 'user'])

        # Save stats to database for caching (optional)
        stats_data = {
            'user_id': user_object_id,
            'total_conversations': total_conversations,
            'urls_scraped': total_scraped_urls,
            'total_messages': total_messages,
            'total_sessions': total_sessions,
            'questions_asked': total_questions,
            'last_updated': datetime.now()
        }

        user_statistics_collection.update_one(
            {'user_id': user_object_id},
            {'$set': stats_data},
            upsert=True
        )

        return jsonify({
            'success': True,
            'statistics': stats_data
        })

    except Exception as e:
        print(f"[ERROR] Failed to get user statistics: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ------------------- START USER SESSION -------------------
@app.route('/start-session', methods=['POST'])
def start_user_session():
    data = request.get_json()
    user_id = data.get('userId')
    
    if not user_id:
        return jsonify({'success': False, 'message': 'User ID is required'}), 400
    
    try:
        user_object_id = ObjectId(user_id)
        
        # End any existing active sessions
        user_sessions_collection.update_many(
            {'user_id': user_object_id, 'is_active': True},
            {'$set': {'is_active': False, 'end_time': datetime.now()}}
        )
        
        # Start new session
        session_data = {
            'user_id': user_object_id,
            'start_time': datetime.now(),
            'is_active': True,
            'timestamp': datetime.now()
        }
        
        result = user_sessions_collection.insert_one(session_data)
        
        return jsonify({
            'success': True,
            'sessionId': str(result.inserted_id),
            'message': 'Session started successfully'
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to start session: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ------------------- END USER SESSION -------------------
@app.route('/end-session', methods=['POST'])
def end_user_session():
    data = request.get_json()
    user_id = data.get('userId')
    session_id = data.get('sessionId')
    
    if not user_id:
        return jsonify({'success': False, 'message': 'User ID is required'}), 400
    
    try:
        user_object_id = ObjectId(user_id)
        
        if session_id:
            # End specific session
            session_object_id = ObjectId(session_id)
            user_sessions_collection.update_one(
                {'_id': session_object_id, 'user_id': user_object_id},
                {'$set': {'is_active': False, 'end_time': datetime.now()}}
            )
        else:
            # End all active sessions for user
            user_sessions_collection.update_many(
                {'user_id': user_object_id, 'is_active': True},
                {'$set': {'is_active': False, 'end_time': datetime.now()}}
            )
        
        return jsonify({
            'success': True,
            'message': 'Session ended successfully'
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to end session: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ------------------- CHANGE PASSWORD -------------------
@app.route('/change-password', methods=['POST'])
def change_password():
    data = request.get_json()
    user_id = data.get('userId')
    current_password = data.get('currentPassword')
    new_password = data.get('newPassword')
    
    if not user_id or not current_password or not new_password:
        return jsonify({'success': False, 'message': 'All fields are required'}), 400
    
    try:
        user_object_id = ObjectId(user_id)
        user = users_collection.find_one({'_id': user_object_id})
        
        if not user:
            return jsonify({'success': False, 'message': 'User not found'}), 404
        
        # Verify current password
        if not check_password_hash(user.get('password'), current_password):
            return jsonify({'success': False, 'message': 'Current password is incorrect'}), 401
        
        # Hash and update new password
        hashed_password = generate_password_hash(new_password)
        users_collection.update_one(
            {'_id': user_object_id},
            {'$set': {'password': hashed_password}}
        )
        
        return jsonify({
            'success': True,
            'message': 'Password updated successfully'
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to change password: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# Helper function to initialize user profile on signup/login
def initialize_user_profile(user_id, user_name, user_email):
    try:
        user_object_id = ObjectId(user_id)
        
        # Check if profile already exists
        existing_profile = user_profiles_collection.find_one({'user_id': user_object_id})
        if existing_profile:
            return
        
        # Create initial profile
        profile_data = {
            'user_id': user_object_id,
            'first_name': user_name.split()[0] if user_name else '',
            'last_name': ' '.join(user_name.split()[1:]) if user_name and len(user_name.split()) > 1 else '',
            'bio': '',
            'location': '',
            'timezone': 'UTC',
            'language': 'English',
            'account_type': 'Free User',
            'is_verified': True,
            'avatar': '',
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        user_profiles_collection.insert_one(profile_data)
        
        # Initialize statistics
        stats_data = {
            'user_id': user_object_id,
            'total_messages': 0,
            'urls_scraped': 0,
            'questions_asked': 0,
            'success_rate': 86,
            'total_sessions': 0,
            'today_time': '0h 0m',
            'created_at': datetime.now(),
            'last_updated': datetime.now()
        }
        
        user_statistics_collection.insert_one(stats_data)
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize user profile: {e}")

def initialize_app():
    """Initialize Docker services, connect to Milvus, and load models with enhanced features."""
    global milvus_collection, sentence_model
    
    print("\n🔌 Connecting to Milvus...")
    try:
        if connect_milvus_safe():
            # Initialize sentence model for dimension calculation
            temp_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
            dim = temp_model.get_sentence_embedding_dimension()
            milvus_collection = ensure_collection_safe(COLLECTION_NAME, dim)
            if milvus_collection:
                print(f"✅ Milvus collection ready: {COLLECTION_NAME}")
            else:
                print("⚠️ Milvus collection creation failed")
        else:
            print("⚠️ Milvus connection failed - continuing without vector search")
            milvus_collection = None
    except Exception as e:
        print(f"⚠️ Milvus setup failed: {e} - continuing without vector search")
        milvus_collection = None

    print("\n🤖 Initializing ML models...")
    init_models()
    
    print("\n✅ Application initialization complete!")
    print("🚀 Server is ready to handle requests!")

@app.route('/toggle-crawler', methods=['POST'])
def toggle_crawler():
    """Endpoint to test crawler functionality"""
    data = request.get_json()
    test_url = data.get('url', 'https://example.com')
    
    try:
        collector = URLCollector(test_url)
        urls = collector.collect()
        
        return jsonify({
            'success': True,
            'message': f'Crawler collected {len(urls)} URLs',
            'urls': urls[:10],  # Return first 10 URLs for preview
            'total_count': len(urls)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Crawler test failed: {str(e)}'
        }), 500

def process_page_data(item):
    """Process individual page data with topic extraction and classification"""
    k, v = item
    try:
        v.pop("cleaned_content", None)
        
        # Extract topic and classify
        if topic_extractor and classifier:
            v["topic"] = topic_extractor.extract_main_topic(v["lemmatized_content"])
            v["predicted_category"] = classifier.classify(v["lemmatized_content"])
        else:
            v["topic"] = "General"
            v["predicted_category"] = "General"
            
        return k, v
    except Exception as e:
        print(f"[ERROR] Processing page {k}: {e}")
        v["topic"] = "General"
        v["predicted_category"] = "General"
        return k, v
    
def ensure_models_ready():
    """Ensure models are initialized before use"""
    if not _models_initialized:
        print("[WARNING] Models not initialized, attempting initialization...")
        return init_models()
    return True

def process_qa_generation_threaded(item):
    """Generate QA pairs for individual page using threading"""
    page_key, info = item
    try:
        topic = info.get("topic", "Unknown")
        category = info.get("predicted_category", "General")
        raw_content = info.get("content", "")
        title = info.get("title", "")
        url = info.get("url", "")

        # Extract contact details and enhance content
        contact_info = extract_contact_info(raw_content)
    
        extra_info = ""
        if contact_info.get("mail"):
            extra_info += "\nEmail(s): " + ", ".join(contact_info["mail"])
        if contact_info.get("phone"):
            extra_info += "\nPhone Number(s): " + ", ".join(contact_info["phone"])
        if contact_info.get("address"):
            extra_info += "\nAddress: " + contact_info["address"]

        # Final content for QA generation
        content = f"{title}\n\n{raw_content.strip()}\n{extra_info.strip()}"

        # Generate QA pairs using the question generator
        qa_pairs = question_generator.generate_questions_and_answers(topic, category, content)
        
        # Ensure qa_pairs is not empty
        if not qa_pairs:
            qa_pairs = [{
                'question': f"What is this page about?",
                'short_answer': f"This page is about {topic}",
                'long_answer': f"This page discusses {topic} in the context of {category}. {content[:200]}..."
            }]
        
        return page_key, {
            "url": url, 
            "title": title, 
            "topic": topic, 
            "qa_pairs": qa_pairs
        }
        
    except Exception as e:
        print(f"❌ Error processing QA for {page_key}: {e}")
        return page_key, {
            "url": info.get("url", ""),
            "title": info.get("title", ""),
            "topic": info.get("topic", "Unknown"),
            "qa_pairs": [{
                'question': f"What can you tell me about this page?",
                'short_answer': "Content processing failed",
                'long_answer': "There was an issue processing this page content."
            }]
        }

def batch_process_with_threading(lemmatized_data, max_workers=5):
    """Process pages using ThreadPoolExecutor for better performance"""
    classified = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_page_data, item) for item in lemmatized_data.items()]
        for future in as_completed(futures):
            try:
                k, v = future.result()
                classified[k] = v
                print(f"🔹 {k} — {v['predicted_category']}")
            except Exception as e:
                print(f"[ERROR] Future failed: {e}")
    return classified

@app.route('/fetch-more', methods=['POST'])
def fetch_more():
    data = request.get_json()
    conv_id = data.get('conversationId')
    if not conv_id:
        return jsonify({'success': False, 'message': 'conversationId required'}), 400
    conv = conversation_collection.find_one({'_id': ObjectId(conv_id)})
    if not conv:
        return jsonify({'success': False, 'message': 'conversation not found'}), 404
    extras = conv.get('pending_extras', [])
    # clear after fetch
    conversation_collection.update_one({'_id': ObjectId(conv_id)}, {'$unset': {'pending_extras': ""}})
    return jsonify({'success': True, 'more_answers': extras}), 200

def split_by_sections(text: str):
    """Split Wikipedia articles by sections instead of raw length"""
    sections = []
    current = []
    for line in text.splitlines():
        if line.strip().startswith("==") and line.strip().endswith("=="):  # Wikipedia heading
            if current:
                sections.append(" ".join(current))
                current = []
        current.append(line.strip())
    if current:
        sections.append(" ".join(current))
    return sections


# Update the main execution block
if __name__ == '__main__':
    try:
        print("🔥 Starting WebscrapQA Backend Server...")
        initialize_app()
        
        # Remove the duplicate init_models() call here
        # init_models()  # ❌ Remove this line
        
        print("🌐 Starting Flask server on http://0.0.0.0:5000")
        socketio.run(app, host="0.0.0.0", port=5000, debug=True,use_reloader=False)

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"💥 Fatal error starting server: {e}")
        traceback.print_exc()
