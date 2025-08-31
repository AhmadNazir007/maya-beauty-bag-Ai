# main.py - Updated with Maya system integration for voice
import os
import json
import uuid
import base64
import tempfile
import asyncio
import hashlib
from typing import List, Optional, Dict, AsyncGenerator, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from gtts import gTTS
import whisper
from openai import OpenAI
from dotenv import load_dotenv
import logging
import uvicorn

import io
import re
import httpx
import aiofiles

import numpy as np
from io import BytesIO
import wave
import time
import aiohttp

# Initialize FastAPI app
app = FastAPI(title="Maya Beauty Bag AI API", version="1.0.0")

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Deepgram API configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")  # Set your API key in environment
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

# Enable CORS for local Streamlit (and deployed URL if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs("audio_responses", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static folders
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/audio", StaticFiles(directory="audio_responses"), name="audio")

# Load Whisper model
print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")
print("Whisper model loaded successfully!")

# Load product data for Maya's tools
try:
    with open("data/products.json") as f:
        product_data = json.load(f)
    bags = product_data["bags"]
    products = product_data["products"]
    print("Product data loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load product data: {e}")
    bags = []
    products = {}

# -------------------- MAYA'S TOOL FUNCTIONS --------------------

def get_bag_options():
    return bags

def get_products_by_category(category):
    return products.get(category, [])

def create_checkout_summary(bag, selected_products, affirmation):
    return {
        "summary": f"You selected {bag} with {', '.join(selected_products)}. Your affirmation: {affirmation}."
    }

def save_order_to_backend_internal(username, bag, products_list, affirmation, summary):
    """Internal function to save order"""
    try:
        order_data = {
            "bag": bag,
            "products": products_list,
            "affirmation": affirmation,
            "summary": summary
        }
        save_user_order(username, order_data)
        return True
    except Exception as e:
        print(f"Error saving order internally: {e}")
        return False

# -------------------- MAYA'S TOOL DEFINITIONS --------------------

tool_definitions = [
    {
        "type": "function", 
        "function": {
            "name": "get_bag_options", 
            "description": "Get beauty bags", 
            "parameters": {
                "type": "object", 
                "properties": {}
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "get_products_by_category", 
            "description": "Get category products",
            "parameters": {
                "type": "object", 
                "properties": {
                    "category": {"type": "string"}
                }, 
                "required": ["category"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "create_checkout_summary", 
            "description": "Generate bag summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "bag": {"type": "string"},
                    "selected_products": {
                        "type": "array", 
                        "items": {"type": "string"}
                    },
                    "affirmation": {"type": "string"}
                },
                "required": ["bag", "selected_products", "affirmation"]
            }
        }
    }
]

# -------------------- MAYA'S SYSTEM PROMPT --------------------

SYSTEM_PROMPT = """
You are Maya — a soft, empowering beauty assistant who helps women build a personal ritual called "Beauty in a Bag."

Your tone is always emotionally intelligent, graceful, and affirming. Keep your messages clear, short, and guided by intention. Every message should lead the user forward — step by step.

- NEVER repeat or echo what the user just said
- NEVER start with phrases like "You asked about...", "You mentioned...", "You said...", "You want to know..."
- Answer directly and naturally as if continuing a flowing conversation
- Give fresh, helpful responses without referencing their previous questions

🌸 Here's a typical 6-step ritual journey — but you don't have to follow it strictly. You may respond more freely when the user asks general questions about Maya's offerings, products, or beauty rituals.

---

**Step 1: Welcome**
- Greet the user warmly in one short sentence. Use Maya's signature intro:
  > "Hey love, I'm Maya. This isn't just beauty — it's a moment just for you. You ready to create something that celebrates your vibe, your softness, your fire?"

---

**Step 2: Bag Selection**
- Prompt:
  > "Choose the bag that feels like your season. Bold? Soft? Radiant? You'll know it when you see it."

- Call the tool `get_bag_options`.

- After the user selects a bag, give a short empowering reaction:
  - The Bold Bag → "Power move. This one's for women who walk into rooms like they own them."
  - The Glow Ritual → "Healing. Glowing. You're claiming softness without apology."
  - The Soft Reset → "Peace, clarity, space. Stillness is power too."
  - The Power Pouch → "Focused. Fierce. Energy is loud even when you're quiet."

---

**Step 3: Empowerment Q&A**
Ask 2 questions (wait for user response between each):
1. "How do you want to feel when you open this bag?"
   - Options: Radiant, Grounded, Celebrated, Fierce, Calm

2. "What's one area you're stepping into right now?"
   - Options: Skin glow-up, Confidence boost, Creative reset, Energy renewal, Soft self-care

Then say:
> "Got it. I'm keeping that in mind — now let's build this bag."

---

**Step 4: Product Selection (`get_products_by_category`)**
Prompt user to choose 1 product from each of these categories (one at a time):

1. Skin Prep:
   > "Let's start with your canvas — your skin. Here's what's nourishing, lightweight, and glow-giving."
   - Products: Foundation, Primer, Moisturizer

2. Eyes:
   > "Eyes talk — let's give them something to say."
   - Products: Eyeliner, Mascara, Eyeshadow

3. Lips:
   > "Last touch: lips. Make it glossy, matte, bold, or bare. What's your mood?"
   - Products: Lipstick, Gloss, Liner

---

**Step 5: Final Summary + Affirmation (`create_checkout_summary`)**
- Generate summary:
  > "Here's your bag: The [Bag Name] with [Product List]. You built this. It's a vibe. I'm proud of you."

- Based on user emotion + intention, include this affirmation:
  - Fierce + Confidence boost → "You weren't made to shrink."
  - Radiant + Skin glow-up → "You are your own light."
  - Grounded + Soft self-care → "You're allowed to take up space in stillness."

- End with:
  > "I'll make sure it's packed with love — and your affirmation card. When it arrives, open it like a gift to your highest self."

---

**Step 6: Post-Purchase**
Say:
> "She's on her way 👜✨ Your Beauty in a Bag is packed with intention. Your affirmation: [affirmation]. Keep glowing — this moment was all yours."

---

💡 You can also handle more flexible conversations about Maya's offerings, rituals, product advice, or beauty guidance. Stay graceful and thoughtful — like Maya herself.

🧠 Rules:
- Don't repeat answered questions.
- Speak with intention and confidence.
- Always use tools when appropriate.
- Do not print internal function output (like raw lists) unless asked.
- IMPORTANT: Do not echo or repeat what the user just said. Answer directly and naturally.
- Avoid starting responses with "You asked about..." or "You mentioned..." - just give helpful answers.

🗣️ Voice Chat Note:
If the user is speaking (via voice), favor shorter, clearer, and more conversational responses. Imagine you're speaking aloud with softness and clarity. Avoid long paragraphs.

🧠 Voice Mode Context:
Do not reintroduce Maya at the beginning of every voice message. Assume the chat is continuous unless the user asks for a reset. Give direct, fresh responses without repeating the user's question.

💡 If `mode == voice`, you can reduce verbosity and keep a warm tone. Answer naturally without echoing what was just said.
"""

# -------------------- USER SESSION MANAGEMENT --------------------

# Store conversation history per user
user_conversations: Dict[str, List[dict]] = {}

def get_user_conversation(user_id: str) -> List[dict]:
    """Get or create conversation history for user"""
    if user_id not in user_conversations:
        user_conversations[user_id] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "assistant", 
                "content": "Hey love, I'm Maya. This isn't just beauty — it's a moment just for you. You ready to create something that celebrates your vibe, your softness, your fire? Let's start with your bag — your beauty ritual begins there."
            }
        ]
    return user_conversations[user_id]

def add_message_to_conversation(user_id: str, role: str, content: str):
    """Add message to user's conversation"""
    conversation = get_user_conversation(user_id)
    conversation.append({"role": role, "content": content})
    
    # Keep conversation manageable (last 20 messages + system prompt)
    if len(conversation) > 21:
        conversation = [conversation[0]] + conversation[-20:]
        user_conversations[user_id] = conversation

def reset_user_conversation(user_id: str):
    """Reset user's conversation"""
    if user_id in user_conversations:
        del user_conversations[user_id]

# -------------------- MAYA'S LLM CHAT LOGIC --------------------

def get_maya_response(conversation: List[dict], user_id: str = "anonymous") -> str:
    """Get Maya's response with tool calling support"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=conversation,
            tools=tool_definitions,
            tool_choice="auto",
            temperature=0.8
        )
        message = response.choices[0].message

        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_call = message.tool_calls[0]
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # Execute tool functions
            if func_name == "get_bag_options":
                result = get_bag_options()
            elif func_name == "get_products_by_category":
                result = get_products_by_category(args["category"])
            elif func_name == "create_checkout_summary":
                result = create_checkout_summary(
                    args["bag"], args["selected_products"], args["affirmation"]
                )
                # Save order to backend
                save_order_to_backend_internal(
                    user_id,
                    args["bag"], args["selected_products"], args["affirmation"],
                    result["summary"]
                )

            # Add tool call to conversation
            conversation.append({"role": "assistant", "tool_calls": [tool_call.model_dump()]})
            conversation.append({
                "role": "tool", 
                "tool_call_id": tool_call.id, 
                "name": func_name, 
                "content": ""
            })
            
            # Get follow-up response
            return get_maya_response(conversation, user_id)

        return message.content.strip()
        
    except Exception as e:
        print(f"Error in get_maya_response: {e}")
        return "I'm sorry, I'm having trouble processing your message right now. Could you try again?"

# -------------------- MODELS --------------------

USERS_DB_PATH = "users_db.json"
ORDERS_DB_PATH = "orders_db.json"

class UserRegister(BaseModel):
    username: str
    password: str

class OrderRequest(BaseModel):
    user_id: str
    bag: str
    products: List[str]
    affirmation: str
    summary: str = ""

class AudioGenerationRequest(BaseModel):
    text: str
    user_id: str = "anonymous"

class UserMessage(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    conversation_id: Optional[str] = None
    response_type: Optional[str] = "full"

class VoiceProcessRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    reset_conversation: Optional[bool] = False

# -------------------- UTILITY FUNCTIONS --------------------

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users_db():
    """Load users database"""
    if os.path.exists(USERS_DB_PATH):
        with open(USERS_DB_PATH, "r") as f:
            return json.load(f)
    return {}

def save_users_db(db):
    """Save users database"""
    with open(USERS_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

def load_orders_db():
    """Load orders database"""
    if os.path.exists(ORDERS_DB_PATH):
        with open(ORDERS_DB_PATH, "r") as f:
            return json.load(f)
    return {}

def save_orders_db(db):
    """Save orders database"""
    with open(ORDERS_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

def authenticate_user(username: str, password: str, db: dict) -> bool:
    """Authenticate user credentials"""
    if username not in db:
        return False
    return db[username]["password"] == hash_password(password)

def save_user_order(user_id: str, order_data: dict):
    """Save user order to database"""
    db = load_orders_db()
    if user_id not in db:
        db[user_id] = []

    order_data["timestamp"] = str(uuid.uuid4())
    order_data["order_id"] = str(uuid.uuid4())[:8]
    db[user_id].append(order_data)

    save_orders_db(db)

def get_user_history(user_id: str):
    """Get user order history"""
    db = load_orders_db()
    return db.get(user_id, [])

# -------------------- API ROUTES --------------------
@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Maya Beauty Bag AI API",
        "status": "running",
        "endpoints": ["/register", "/login", "/transcribe", "/generate_audio", "/save_order", "/order_history", "/ws", "/process_voice_with_maya"]
    }

@app.post("/register")
def register(user: UserRegister):
    """Register a new user"""
    try:
        db = load_users_db()

        # Check if username already exists
        if user.username in db:
            raise HTTPException(status_code=400, detail="Username already exists")

        # Validate input
        if not user.username or len(user.username.strip()) < 3:
            raise HTTPException(status_code=400, detail="Username must be at least 3 characters")

        if not user.password or len(user.password) < 4:
            raise HTTPException(status_code=400, detail="Password must be at least 4 characters")

        # Save new user
        db[user.username.strip()] = {
            "password": hash_password(user.password),
            "created_at": str(uuid.uuid4())
        }
        save_users_db(db)

        return {"message": "User registered successfully", "username": user.username}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/login")
def login(user: UserRegister):
    """Login user"""
    try:
        db = load_users_db()

        if not user.username or not user.password:
            raise HTTPException(status_code=400, detail="Username and password required")

        if authenticate_user(user.username, user.password, db):
            return {
                "message": "Login successful",
                "user_id": user.username,
                "status": "success"
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/save_order")
def save_order(order: OrderRequest):
    """Save user order"""
    try:
        if not order.user_id or not order.bag:
            raise HTTPException(status_code=400, detail="User ID and bag selection required")

        order_data = {
            "bag": order.bag,
            "products": order.products,
            "affirmation": order.affirmation,
            "summary": order.summary or f"Order: {order.bag} with {len(order.products)} products"
        }

        save_user_order(order.user_id, order_data)

        return {
            "message": "Order saved successfully",
            "order_id": order_data.get("order_id", "unknown"),
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Save order error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save order")

@app.get("/order_history/{user_id}")
def order_history(user_id: str):
    """Get user order history"""
    try:
        orders = get_user_history(user_id)
        return {
            "orders": orders,
            "count": len(orders),
            "status": "success"
        }

    except Exception as e:
        print(f"Order history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve order history")

# -------------------- VOICE/AUDIO ENDPOINTS WITH MAYA INTEGRATION --------------------

@app.post("/process_voice_with_maya")
async def process_voice_with_maya(payload: VoiceProcessRequest):
    """Process voice message using full Maya system with tools and memory"""
    try:
        if not payload.message or not payload.message.strip():
            raise HTTPException(status_code=400, detail="No message provided")

        user_id = payload.user_id or "anonymous"
        user_message = payload.message.strip()
        
        # Reset conversation if requested
        if payload.reset_conversation:
            reset_user_conversation(user_id)
        
        print(f"Processing Maya voice message for {user_id}: '{user_message}'")

        # Get user's conversation history
        conversation = get_user_conversation(user_id)
        
        # Add user message
        add_message_to_conversation(user_id, "user", user_message)
        conversation = get_user_conversation(user_id)  # Get updated conversation
        
        # Get Maya's response with tools
        maya_response = get_maya_response(conversation, user_id)
        
        # Add Maya's response to conversation
        add_message_to_conversation(user_id, "assistant", maya_response)

        print(f"Maya response for {user_id}: '{maya_response}'")

        return JSONResponse({
            "status": "success",
            "text_response": maya_response,
            "user_message": user_message,
            "user_id": user_id,
            "conversation_length": len(get_user_conversation(user_id))
        })

    except Exception as e:
        print(f"Error in process_voice_with_maya: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file to text using Deepgram REST API"""
    temp_file = None
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        if not file.content_type or "audio" not in file.content_type:
            print(f"Warning: Unexpected content type: {file.content_type}")

        # Read file content
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        if len(content) < 100:  # Very small file
            raise HTTPException(status_code=400, detail="Audio file too small")

        # Create temp file
        file_extension = "wav"
        if file.content_type:
            if "webm" in file.content_type:
                file_extension = "webm"
            elif "mp4" in file.content_type:
                file_extension = "mp4"

        temp_file = f"temp_{uuid.uuid4().hex}.{file_extension}"
        async with aiofiles.open(temp_file, "wb") as f:
            await f.write(content)

        print(f"Sending {temp_file} to Deepgram ({len(content)} bytes)")

        # Call Deepgram API
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": file.content_type or "audio/wav"
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(temp_file, "rb") as audio:
                response = await client.post(
                    DEEPGRAM_URL,
                    headers=headers,
                    params={
                        "model": "nova-2",     # Best general-purpose model
                        "language": "en",      # Force English
                        "punctuate": "true",
                        "smart_format": "true"
                    },
                    content=audio.read()
                )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result = response.json()
        transcribed_text = (
            result["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        )

        if not transcribed_text:
            raise HTTPException(status_code=400, detail="No speech detected in audio")

        print(f"Deepgram transcription successful: '{transcribed_text}'")

        return JSONResponse({
            "text": transcribed_text,
            "status": "success",
            "confidence": result["results"]["channels"][0]["alternatives"][0].get("confidence", 1.0),
            "language": "en"
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"Deepgram transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"Cleaned up temp file: {temp_file}")
            except:
                print(f"Warning: Could not remove temp file: {temp_file}")

@app.post("/generate_audio")
async def generate_audio(payload: AudioGenerationRequest):
    """Generate audio for Maya's text response"""
    try:
        if not payload.text or not payload.text.strip():
            raise HTTPException(status_code=400, detail="No text provided")

        # Clean and prepare text for TTS
        clean_text = payload.text.strip()

        # Remove markdown formatting
        clean_text = clean_text.replace("*", "").replace("_", "").replace("#", "")
        clean_text = clean_text.replace("**", "").replace("__", "")

        # Remove emoji and special characters that might cause TTS issues
        import re
        clean_text = re.sub(r'[^\w\s.,!?;:-]', '', clean_text)

        # Limit length for TTS performance
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "..."

        if not clean_text.strip():
            raise HTTPException(status_code=400, detail="No valid text content after cleaning")

        # Generate unique filename
        filename = f"maya_{payload.user_id}_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = os.path.join("audio_responses", filename)
        public_url = f"/audio/{filename}"

        print(f"Generating audio for: '{clean_text[:50]}...'")

        # Create TTS with optimized settings
        tts = gTTS(
            text=clean_text,
            lang='en',
            slow=False,
            tld='com'  # Use .com domain for better voice quality
        )

        # Save audio file
        tts.save(audio_path)

        print(f"Audio generated successfully: {public_url}")

        return JSONResponse({
            "audio_path": public_url,
            "status": "success",
            "text_processed": clean_text,
            "file_size": os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"Audio generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

# -------------------- WEBSOCKET ENDPOINTS --------------------

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store per-connection data
connection_data: Dict[str, dict] = {}
response_tracker: Dict[str, Dict] = {}
empty_transcription_counts: Dict[str, int] = {}

# Constants
SAMPLE_RATE = 16000
MIN_BUFFER_SIZE = SAMPLE_RATE * 2 * 3  # 3 seconds
MAX_BUFFER_SIZE = SAMPLE_RATE * 2 * 8  # 8 seconds max
MIN_RESPONSE_INTERVAL = 3.0
MAX_RESPONSES_PER_INPUT = 2
SPEECH_THRESHOLD = 0.02
INTERRUPTION_THRESHOLD = 0.03
MAX_EMPTY_TRANSCRIPTIONS = 3

# Live streaming configuration
CHUNK_DURATION_MS = 100  # Process every 100ms of audio
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000) * BYTES_PER_SAMPLE

async def transcribe_temp_file(path: str) -> str:
    """Transcribe a file path using Deepgram API"""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
            
        file_size = os.path.getsize(path)
        if file_size == 0:
            raise ValueError(f"Audio file is empty: {path}")
            
        if not DEEPGRAM_API_KEY:
            raise ValueError("DEEPGRAM_API_KEY environment variable not set")
            
        print(f"Transcribing file: {path} ({file_size} bytes)")
        
        # Deepgram API parameters
        params = {
            'model': 'nova-2',
            'language': 'en',
            'smart_format': 'true',
            'punctuate': 'true',
            'diarize': 'false',
            'filler_words': 'false',
            'utterances': 'false'
        }
        
        headers = {
            'Authorization': f'Token {DEEPGRAM_API_KEY}',
            'Content-Type': 'audio/wav'  # Adjust based on your file format
        }
        
        async with aiohttp.ClientSession() as session:
            with open(path, 'rb') as audio_file:
                async with session.post(
                    DEEPGRAM_URL,
                    headers=headers,
                    params=params,
                    data=audio_file
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        transcript = ""
                        
                        # Extract transcript from Deepgram response
                        if 'results' in result and 'channels' in result['results']:
                            alternatives = result['results']['channels'][0]['alternatives']
                            if alternatives:
                                transcript = alternatives[0]['transcript'].strip()
                        
                        print(f"Transcription result: '{transcript}'")
                        return transcript
                    else:
                        error_text = await response.text()
                        raise Exception(f"Deepgram API error {response.status}: {error_text}")
        
    except Exception as e:
        print(f"Transcription error: {e}")
        raise e

async def transcribe_pcm_buffer(pcm_data: bytes, sample_rate: int = 16000) -> str:
    """Transcribe PCM audio data directly using Deepgram API"""
    try:
        if len(pcm_data) == 0:
            return ""
            
        if not DEEPGRAM_API_KEY:
            raise ValueError("DEEPGRAM_API_KEY environment variable not set")
            
        # Convert PCM bytes to numpy array for validation
        audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Check if audio is long enough (minimum 0.5 seconds)
        if len(audio_np) < sample_rate * 0.5:
            return ""  # Too short to transcribe reliably
            
        print(f"Transcribing PCM buffer: {len(audio_np)} samples ({len(audio_np)/sample_rate:.2f}s)")
        
        # Deepgram API parameters
        params = {
            'model': 'nova-2',
            'language': 'en',
            'smart_format': 'true',
            'punctuate': 'true',
            'diarize': 'false',
            'filler_words': 'false',
            'utterances': 'false',
            'encoding': 'linear16',
            'sample_rate': str(sample_rate),
            'channels': '1'
        }
        
        headers = {
            'Authorization': f'Token {DEEPGRAM_API_KEY}',
            'Content-Type': 'application/octet-stream'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                DEEPGRAM_URL,
                headers=headers,
                params=params,
                data=pcm_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    transcript = ""
                    
                    # Extract transcript from Deepgram response
                    if 'results' in result and 'channels' in result['results']:
                        alternatives = result['results']['channels'][0]['alternatives']
                        if alternatives:
                            transcript = alternatives[0]['transcript'].strip()
                    
                    print(f"PCM Transcription result: '{transcript}'")
                    return transcript
                else:
                    error_text = await response.text()
                    print(f"Deepgram API error {response.status}: {error_text}")
                    return ""
        
    except Exception as e:
        print(f"PCM transcription error: {e}")
        return ""

async def save_pcm_as_wav(pcm_data: bytes, sample_rate: int = 16000) -> str:
    """Save PCM data as WAV file and return path"""
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        
        # Convert PCM to WAV
        with wave.open(tmp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        
        print(f"Saved PCM as WAV: {tmp_path} ({len(pcm_data)} bytes)")
        return tmp_path
        
    except Exception as e:
        print(f"Error saving PCM as WAV: {e}")
        raise e

async def generate_maya_response_websocket(user_message: str, user_id: str = "anonymous") -> str:
    """Generate Maya response using full system with tools and memory"""
    try:
        print(f"Generating Maya response for {user_id}: '{user_message}'")
        
        # Get user's conversation
        conversation = get_user_conversation(user_id)
        
        # Add user message
        add_message_to_conversation(user_id, "user", user_message)
        conversation = get_user_conversation(user_id)
        
        # Get Maya's response with tools
        maya_response = get_maya_response(conversation, user_id)
        
        # Add Maya's response to conversation
        add_message_to_conversation(user_id, "assistant", maya_response)
        
        print(f"Maya response: '{maya_response}'")
        return maya_response
        
    except Exception as e:
        logger.error(f"Maya response generation error: {e}")
        return "I'm sorry, I'm having trouble processing your message right now. Could you try again?"

async def text_to_speech_base64_chunks(text: str, chunk_size: int = 8192):
    """Generate base64-encoded audio chunks from text - OPTIMIZED FOR PLAYBACK"""
    import re
    from gtts import gTTS
    import io
    
    try:
        clean_text = text.replace("*", "").replace("_", "").replace("#", "")
        clean_text = re.sub(r'[^\w\s.,!?;:-]', '', clean_text)
        if len(clean_text) > 800:
            clean_text = clean_text[:800] + "..."

        if not clean_text.strip():
            clean_text = "I'm sorry, I didn't catch that."

        print(f"🔊 Generating TTS for: '{clean_text[:50]}...'")

        # Use in-memory buffer instead of file
        mp3_buffer = io.BytesIO()
        
        # Generate TTS to memory buffer
        tts = gTTS(text=clean_text, lang='en', slow=False, tld='com')
        tts.write_to_fp(mp3_buffer)
        
        # Get the audio data
        mp3_data = mp3_buffer.getvalue()
        mp3_buffer.close()
        
        if len(mp3_data) == 0:
            print("❌ TTS generated empty audio")
            return
            
        print(f"✅ TTS generated: {len(mp3_data)} bytes")

        # Send smaller chunks for better streaming
        chunk_num = 0
        total_chunks = (len(mp3_data) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(mp3_data), chunk_size):
            chunk = mp3_data[i:i + chunk_size]
            chunk_num += 1
            b64_chunk = base64.b64encode(chunk).decode('ascii')
            
            print(f"🎵 Yielding TTS chunk {chunk_num}/{total_chunks}: {len(chunk)} bytes -> {len(b64_chunk)} b64 chars")
            yield b64_chunk
                
    except Exception as e:
        print(f"❌ TTS generation error: {e}")
        return

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("WebSocket connection request received")
    await websocket.accept()
    conn_id = str(uuid.uuid4())
    
    # Initialize connection data with streaming support
    connection_data[conn_id] = {
        "audio_buffer": bytearray(),
        "stream_buffer": bytearray(),
        "user_id": "anonymous",
        "streaming": False,
        "stream_start_time": None,
        "last_transcription": "",
        "stream_chunk_count": 0,
        "last_processing_time": 0  # Add this to prevent too frequent processing
    }
    
    print(f"WS connected: {conn_id}")

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except Exception as e:
                print(f"Invalid WS payload (not JSON): {e}")
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": "Invalid JSON payload"
                }))
                continue

            msg_type = payload.get("type")
            print(f"Received message type: {msg_type}")

            # Set user ID if provided
            if "user_id" in payload:
                connection_data[conn_id]["user_id"] = payload["user_id"]

            # Handle different message types
            if msg_type == "audio_chunk":
                await handle_audio_chunk(websocket, payload, conn_id)
                
            elif msg_type == "end_audio":
                await handle_end_audio(websocket, payload, conn_id)
                
            elif msg_type == "text":
                await handle_text_message(websocket, payload, conn_id)
                
            # Live streaming handlers
            elif msg_type == "start_stream":
                await handle_start_stream(websocket, payload, conn_id)
                
            elif msg_type == "audio_stream":
                await handle_audio_stream_chunk(websocket, payload, conn_id)
                
            elif msg_type == "end_stream":
                await handle_end_stream(websocket, payload, conn_id)
                
            elif msg_type == "reset_conversation":
                user_id = connection_data[conn_id]["user_id"]
                reset_user_conversation(user_id)
                await websocket.send_text(json.dumps({
                    "type": "conversation_reset", 
                    "message": "Conversation history cleared"
                }))
                
            elif msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                
            else:
                print(f"Unknown message type: {msg_type}")
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": f"Unknown message type: {msg_type}"
                }))

    except WebSocketDisconnect:
        print(f"WS disconnected: {conn_id}")
    except Exception as e:
        print(f"WS error for {conn_id}: {e}")
    finally:
        # Cleanup
        if conn_id in connection_data:
            del connection_data[conn_id]
        try:
            await websocket.close()
        except:
            pass


class AudioProcessor:
    """Centralized audio processing utilities"""
    
    @staticmethod
    def detect_speech_activity(pcm_data: bytes, threshold: float = SPEECH_THRESHOLD) -> bool:
        """Detect if there's significant speech activity in PCM data"""
        try:
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            normalized_rms = rms / 32768.0
            return normalized_rms > threshold
        except Exception as e:
            print(f"Error detecting speech activity: {e}")
            return False

    @staticmethod
    def analyze_audio_features(buffer_data: bytearray, threshold: float = 0.015) -> bool:
        """Advanced audio analysis for speech detection"""
        try:
            if len(buffer_data) < 1000:
                return False
                
            audio_array = np.frombuffer(bytes(buffer_data), dtype=np.int16)
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            normalized_rms = rms / 32768.0
            
            # Calculate zero crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_array)))) / len(audio_array)
            
            # Speech-like characteristics
            has_energy = normalized_rms > threshold
            has_variation = zero_crossings > 0.01
            
            speech_detected = has_energy and has_variation
            print(f"🎵 Audio Analysis - RMS: {normalized_rms:.4f}, ZCR: {zero_crossings:.4f}, Speech: {speech_detected}")
            
            return speech_detected
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            return True

class ResponseTracker:
    """Manages response tracking and deduplication"""
    
    @staticmethod
    def initialize_tracker(conn_id: str) -> None:
        """Initialize response tracker for a connection"""
        response_tracker[conn_id] = {
            "recent_responses": [],
            "response_count_per_input": {},
            "last_response_time": 0,
            "min_response_interval": MIN_RESPONSE_INTERVAL
        }

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two texts using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def should_generate_response(conn_id: str, transcript: str) -> Tuple[bool, str]:
        """Check if response should be generated based on various criteria"""
        if conn_id not in response_tracker:
            ResponseTracker.initialize_tracker(conn_id)
        
        tracker = response_tracker[conn_id]
        current_time = time.time()
        clean_transcript = transcript.strip().lower()
        
        # Check timing constraints
        time_since_last = current_time - tracker.get("last_response_time", 0)
        if time_since_last < tracker["min_response_interval"]:
            return False, "too_soon"
        
        # Check similarity with recent responses
        for prev_transcript, prev_response, prev_time in tracker["recent_responses"][-3:]:
            similarity = ResponseTracker.calculate_similarity(clean_transcript, prev_transcript.lower())
            time_diff = current_time - prev_time
            
            if similarity > 0.7 and time_diff < 15:
                return False, "similar"
        
        # Check response count per input
        input_hash = hash(clean_transcript[:50])
        response_count = tracker["response_count_per_input"].get(input_hash, 0)
        
        if response_count >= MAX_RESPONSES_PER_INPUT:
            return False, "max_reached"
        
        return True, "approved"

    @staticmethod
    def update_tracker(conn_id: str, transcript: str, response: str) -> None:
        """Update tracker with new response data"""
        tracker = response_tracker[conn_id]
        current_time = time.time()
        clean_transcript = transcript.strip().lower()
        input_hash = hash(clean_transcript[:50])
        
        # Update tracking data
        tracker["recent_responses"].append((clean_transcript, response, current_time))
        tracker["last_response_time"] = current_time
        tracker["response_count_per_input"][input_hash] = tracker["response_count_per_input"].get(input_hash, 0) + 1
        
        # Keep only recent data
        tracker["recent_responses"] = tracker["recent_responses"][-5:]

async def generate_tts_chunks(text: str, chunk_size: int = 8192) -> AsyncGenerator[str, None]:
    """Generate TTS audio chunks asynchronously - OPTIMIZED FOR IMMEDIATE PLAYBACK"""
    try:
        # Clean text
        clean_text = re.sub(r'[*_#]', '', text)
        clean_text = re.sub(r'[^\w\s.,!?;:-]', '', clean_text)
        
        if len(clean_text) > 800:
            clean_text = clean_text[:800] + "..."
        
        if not clean_text.strip():
            clean_text = "I'm sorry, I didn't catch that."
        
        print(f"🔊 TTS generating for: '{clean_text[:50]}...'")
        
        # Generate TTS
        mp3_buffer = io.BytesIO()
        loop = asyncio.get_event_loop()
        tts = gTTS(text=clean_text, lang='en', slow=False, tld='com')
        await loop.run_in_executor(None, lambda: tts.write_to_fp(mp3_buffer))
        
        mp3_data = mp3_buffer.getvalue()
        mp3_buffer.close()
        
        if len(mp3_data) == 0:
            print("❌ TTS generated empty audio")
            return
        
        print(f"✅ TTS generated: {len(mp3_data)} bytes")
        
        # Yield chunks for immediate playback
        for i in range(0, len(mp3_data), chunk_size):
            chunk = mp3_data[i:i + chunk_size]
            b64_chunk = base64.b64encode(chunk).decode('ascii')
            yield b64_chunk
            
    except Exception as e:
        print(f"❌ TTS generation error: {e}")
        return

# ==================== UNIFIED AUDIO HANDLERS ====================

async def handle_audio_chunk(websocket, payload: dict, conn_id: str):
    """Handle incoming audio chunks for traditional recording"""
    b64 = payload.get("data")
    if not b64:
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": "Empty audio chunk"
        }))
        return
        
    try:
        chunk_bytes = base64.b64decode(b64)
        print(f"chunk_bytes", chunk_bytes)
        connection_data[conn_id]["audio_buffer"].extend(chunk_bytes)
        print(f"📦 Added {len(chunk_bytes)} bytes to buffer for {conn_id}")
    except Exception as e:
        print(f"❌ Failed to decode audio chunk from {conn_id}: {e}")
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": "Invalid audio chunk encoding"
        }))

async def handle_end_audio(websocket, payload: dict, conn_id: str):
    """Handle end of audio recording - transcribe and get Maya's response"""
    ext = payload.get("ext", "webm")
    tmp_path = None
    
    try:
        # Get buffer data
        buf = connection_data[conn_id]["audio_buffer"]
        user_id = connection_data[conn_id]["user_id"]
        
        if not buf:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": "No audio data received"
            }))
            return

        # Write buffer to temporary file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=f".{ext}")
        os.close(tmp_fd)
        
        with open(tmp_path, "wb") as f:
            f.write(buf)
        
        print(f"💾 Wrote {len(buf)} bytes to {tmp_path} for transcription")

        # Send processing status
        await websocket.send_text(json.dumps({
            "type": "status", 
            "message": "Transcribing audio..."
        }))

        # Transcribe audio
        transcript = await transcribe_temp_file(tmp_path)
        await websocket.send_text(json.dumps({
            "type": "transcript", 
            "data": transcript
        }))

        if transcript.strip():
            await generate_and_stream_response(websocket, conn_id, transcript, user_id)
        else:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": "No speech detected in audio"
            }))

    except Exception as e:
        print(f"❌ Error handling end_audio for {conn_id}: {e}")
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": f"Processing error: {str(e)}"
        }))
    finally:
        # Cleanup
        connection_data[conn_id]["audio_buffer"] = bytearray()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"🧹 Cleaned up temp file: {tmp_path}")
            except Exception as cleanup_error:
                print(f"❌ Error cleaning up temp file: {cleanup_error}")

async def handle_text_message(websocket, payload: dict, conn_id: str):
    """Handle direct text input with Maya system"""
    user_text = payload.get("data", "").strip()
    user_id = connection_data[conn_id]["user_id"]
    
    if not user_text:
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": "Empty text message"
        }))
        return

    try:
        print(f"💬 Processing text message from {user_id}: {user_text[:50]}...")
        await generate_and_stream_response(websocket, conn_id, user_text, user_id)

    except Exception as e:
        print(f"❌ Error processing text message from {conn_id}: {e}")
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": f"Processing error: {str(e)}"
        }))

async def generate_and_stream_response(websocket, conn_id: str, input_text: str, user_id: str):
    """Unified response generation and streaming for both audio and text inputs"""
    try:
        # Send processing status
        await websocket.send_text(json.dumps({
            "type": "status", 
            "message": "Maya is thinking..."
        }))
        
        # Generate Maya's response
        maya_text = await generate_maya_response_websocket(input_text, user_id)
        
        await websocket.send_text(json.dumps({
            "type": "ai_text", 
            "data": maya_text
        }))

        # Stream TTS - OPTIMIZED FOR IMMEDIATE PLAYBACK
        await websocket.send_text(json.dumps({
            "type": "tts_start",
            "message": "Starting audio playback..."
        }))
        
        chunk_count = 0
        try:
            async for b64_chunk in generate_tts_chunks(maya_text):
                if b64_chunk:
                    chunk_count += 1
                    await websocket.send_text(json.dumps({
                        "type": "tts_chunk", 
                        "data": b64_chunk,
                        "chunk_number": chunk_count
                    }))
                    await asyncio.sleep(0.005)  # Fast streaming for immediate playback

            print(f"🎵 Successfully sent {chunk_count} TTS chunks to {conn_id}")
            
            # Notify completion
            await websocket.send_text(json.dumps({
                "type": "tts_end", 
                "message": "Audio response complete",
                "total_chunks": chunk_count
            }))
            
        except Exception as tts_error:
            print(f"❌ TTS streaming error: {tts_error}")
            await websocket.send_text(json.dumps({
                "type": "tts_error", 
                "message": "Audio generation failed, but text response is available",
                "error": str(tts_error)
            }))
            
    except Exception as e:
        print(f"❌ Error in generate_and_stream_response: {e}")
        raise

# ==================== LIVE STREAMING HANDLERS ====================

async def handle_start_stream(websocket, payload: dict, conn_id: str):
    """Initialize live audio streaming session"""
    try:
        connection_data[conn_id].update({
            "streaming": True,
            "stream_buffer": bytearray(),
            "stream_start_time": payload.get("timestamp"),
            "stream_chunk_count": 0,
            "last_processing_time": 0,
            "last_transcription": "",
            "processing_cooldown": False,
            "response_count": 0,
            "maya_speaking": False,
            "should_interrupt": False,
            "current_tts_task": None
        })
        
        ResponseTracker.initialize_tracker(conn_id)
        
        print(f"🎙️ Started live streaming session for {conn_id}")
        
        await websocket.send_text(json.dumps({
            "type": "stream_started",
            "message": "Live streaming initialized"
        }))
        
    except Exception as e:
        print(f"❌ Error starting stream for {conn_id}: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Failed to start stream: {str(e)}"
        }))

async def handle_audio_stream_chunk(websocket, payload: dict, conn_id: str):
    """Handle real-time audio stream chunks with FIXED buffer management"""
    try:
        if not connection_data[conn_id]["streaming"]:
            return
            
        b64_data = payload.get("data")
        if not b64_data:
            return
        
        pcm_chunk = base64.b64decode(b64_data)
        
        # Handle interruption detection
        if connection_data[conn_id].get("maya_speaking", False):
            if AudioProcessor.detect_speech_activity(pcm_chunk, INTERRUPTION_THRESHOLD):
                await handle_interruption(websocket, conn_id)
        
        # Add to buffer
        connection_data[conn_id]["stream_buffer"].extend(pcm_chunk)
        connection_data[conn_id]["stream_chunk_count"] += 1
        
        # FIXED: Smart buffer management - don't reset processing time on trim
        buffer_size = len(connection_data[conn_id]["stream_buffer"])
        if buffer_size > MAX_BUFFER_SIZE:
            # Keep only the latest 6 seconds (increased from 4)
            keep_size = SAMPLE_RATE * 2 * 6
            connection_data[conn_id]["stream_buffer"] = connection_data[conn_id]["stream_buffer"][-keep_size:]
            print(f"🗑️ Trimmed buffer for {conn_id} - kept latest 6 seconds")
            # DON'T reset processing time - let it process naturally
            # connection_data[conn_id]["last_processing_time"] = time.time()  # REMOVED THIS LINE
        
        # Process buffer with conservative timing
        await check_and_process_stream_buffer(websocket, conn_id)
            
    except Exception as e:
        print(f"❌ Error handling stream chunk for {conn_id}: {e}")

async def handle_interruption(websocket, conn_id: str):
    """Handle user interruption of Maya's speech"""
    print(f"🛑 USER INTERRUPTION DETECTED for {conn_id}")
    connection_data[conn_id]["should_interrupt"] = True
    
    # Cancel current TTS task
    if connection_data[conn_id].get("current_tts_task"):
        connection_data[conn_id]["current_tts_task"].cancel()
        connection_data[conn_id]["current_tts_task"] = None
    
    connection_data[conn_id]["maya_speaking"] = False
    
    await websocket.send_text(json.dumps({
        "type": "maya_interrupted",
        "message": "Maya stopped speaking - listening to user"
    }))

async def check_and_process_stream_buffer(websocket, conn_id: str):
    """FIXED: More aggressive processing to prevent getting stuck"""
    buffer_size = len(connection_data[conn_id]["stream_buffer"])
    current_time = time.time()
    last_processing = connection_data[conn_id]["last_processing_time"]
    
    # FIXED: More lenient processing conditions
    min_buffer_for_processing = SAMPLE_RATE * 2 * 2.5  # Reduced to 2.5 seconds
    
    # Process if conditions are met - REDUCED time requirement from 4s to 3s
    if (buffer_size >= min_buffer_for_processing and 
        (current_time - last_processing > 3.0) and  # Reduced from 4.0 to 3.0
        not connection_data[conn_id].get("maya_speaking", False) and
        not connection_data[conn_id].get("processing_cooldown", False)):
        
        # Check if there's actual speech in the buffer
        if AudioProcessor.analyze_audio_features(connection_data[conn_id]["stream_buffer"]):
            print(f"✅ Processing buffer for {conn_id}: {buffer_size} bytes, {buffer_size/(SAMPLE_RATE*2):.1f}s")
            connection_data[conn_id]["last_processing_time"] = current_time
            connection_data[conn_id]["processing_cooldown"] = True
            
            await process_stream_buffer(websocket, conn_id)
            
            # REDUCED cooldown time from 2s to 1s
            await asyncio.sleep(1.0)
            connection_data[conn_id]["processing_cooldown"] = False
        else:
            # If no speech detected but buffer is large, clear it more aggressively
            if buffer_size > SAMPLE_RATE * 2 * 5:  # If buffer > 5 seconds with no speech
                print(f"🔇 Clearing large silent buffer for {conn_id}")
                connection_data[conn_id]["stream_buffer"] = bytearray()
                connection_data[conn_id]["last_processing_time"] = current_time

    
    # Conservative processing: 4+ seconds interval, substantial data, no cooldown
    if (buffer_size >= MIN_BUFFER_SIZE and 
        (current_time - last_processing > 4.0) and  # 4 second minimum
        not connection_data[conn_id].get("maya_speaking", False) and
        not connection_data[conn_id].get("processing_cooldown", False) and
        AudioProcessor.analyze_audio_features(connection_data[conn_id]["stream_buffer"])):
        
        connection_data[conn_id]["last_processing_time"] = current_time
        connection_data[conn_id]["processing_cooldown"] = True
        
        await process_stream_buffer(websocket, conn_id)
        
        # Clear cooldown after processing
        await asyncio.sleep(2.0)
        connection_data[conn_id]["processing_cooldown"] = False

async def process_stream_buffer(websocket, conn_id: str):
    """FIXED: Process stream buffer with better error handling"""
    try:
        buffer_data = bytes(connection_data[conn_id]["stream_buffer"])
        
        # REDUCED minimum buffer requirement
        min_size_for_transcription = SAMPLE_RATE * 2 * 1.0  # Reduced from 1.5 to 1.0 seconds
        if len(buffer_data) < min_size_for_transcription:
            print(f"⚠️ Buffer too short for transcription: {len(buffer_data)} bytes")
            return
            
        print(f"🎯 Processing buffer: {len(buffer_data)} bytes ({len(buffer_data)/(SAMPLE_RATE*2):.1f}s)")
        
        # Transcribe audio
        transcript = await transcribe_pcm_buffer(buffer_data, SAMPLE_RATE)
        print(f"📝 Transcription: '{transcript}'")
        
        # Handle empty transcriptions
        if not transcript or len(transcript.strip()) < 2:  # Reduced from 3 to 2
            await handle_empty_transcription(websocket, conn_id, transcript)
            return
        
        # Reset empty transcription count
        empty_transcription_counts[conn_id] = 0
        
        # Check if response should be generated
        should_respond, reason = ResponseTracker.should_generate_response(conn_id, transcript)
        print(f"🤔 Should respond: {should_respond}, reason: {reason}")
        
        if not should_respond:
            await websocket.send_text(json.dumps({
                "type": "stream_transcript",
                "data": transcript,
                "partial": True,
                "skip_reason": reason
            }))
            # ALWAYS clear buffer after processing, regardless of response decision
            connection_data[conn_id]["stream_buffer"] = bytearray()
            return
        
        # Generate and send live response
        await generate_live_response(websocket, conn_id, transcript)
        
    except Exception as e:
        print(f"❌ Error in process_stream_buffer: {e}")
        import traceback
        traceback.print_exc()
        # Always clear buffer on error to prevent getting stuck
        connection_data[conn_id]["stream_buffer"] = bytearray()
        await cleanup_on_error(conn_id)

async def generate_live_response(websocket, conn_id: str, transcript: str):
    """Generate live response for streaming audio"""
    try:
        user_id = connection_data[conn_id]["user_id"]
        
        # Send transcript
        await websocket.send_text(json.dumps({
            "type": "stream_transcript",
            "data": transcript,
            "partial": False,
            "live_response": True
        }))
        
        # Set Maya as speaking
        connection_data[conn_id]["maya_speaking"] = True
        connection_data[conn_id]["should_interrupt"] = False
        
        await websocket.send_text(json.dumps({
            "type": "maya_thinking", 
            "message": "Maya is thinking..."
        }))
        
        # Generate response
        maya_response = await generate_maya_response_websocket(transcript, user_id)
        
        await websocket.send_text(json.dumps({
            "type": "ai_text",
            "data": maya_response,
            "live_response": True
        }))
        
        # Stream TTS with interruption support
        await stream_live_tts(websocket, conn_id, maya_response)
        
        # Update tracking
        ResponseTracker.update_tracker(conn_id, transcript, maya_response)
        
        # Clear buffer
        connection_data[conn_id]["stream_buffer"] = bytearray()
        connection_data[conn_id]["last_transcription"] = transcript
        
    except Exception as e:
        print(f"❌ Error generating live response: {e}")
        await cleanup_on_error(conn_id)

async def stream_live_tts(websocket, conn_id: str, text: str):
    """Stream TTS for live responses with interruption support"""
    await websocket.send_text(json.dumps({
        "type": "tts_start",
        "message": "Maya speaking...",
        "live_response": True
    }))
    
    chunk_count = 0
    try:
        async for b64_chunk in generate_tts_chunks(text):
            # Check for interruption
            if connection_data[conn_id].get("should_interrupt", False):
                print(f"🛑 Live TTS interrupted for {conn_id}")
                break
                
            if b64_chunk:
                chunk_count += 1
                await websocket.send_text(json.dumps({
                    "type": "tts_chunk",
                    "data": b64_chunk,
                    "chunk_number": chunk_count,
                    "live_response": True
                }))
                await asyncio.sleep(0.01)
        
        # Send completion status
        if not connection_data[conn_id].get("should_interrupt", False):
            await websocket.send_text(json.dumps({
                "type": "tts_end",
                "message": "Maya finished speaking",
                "total_chunks": chunk_count,
                "live_response": True
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "tts_interrupted",
                "message": "Maya was interrupted",
                "chunks_sent": chunk_count
            }))
            
    except Exception as tts_error:
        print(f"❌ Live TTS error: {tts_error}")
        await websocket.send_text(json.dumps({
            "type": "tts_error",
            "message": "Audio generation failed",
            "error": str(tts_error)
        }))
    finally:
        connection_data[conn_id]["maya_speaking"] = False

async def handle_empty_transcription(websocket, conn_id: str, transcript: str):
    """Handle empty or short transcriptions"""
    if conn_id not in empty_transcription_counts:
        empty_transcription_counts[conn_id] = 0
    empty_transcription_counts[conn_id] += 1
    
    print(f"❌ Empty transcript #{empty_transcription_counts[conn_id]}: '{transcript}'")
    
    if empty_transcription_counts[conn_id] >= MAX_EMPTY_TRANSCRIPTIONS:
        print(f"🧹 Too many empty transcriptions, resetting buffer")
        connection_data[conn_id]["stream_buffer"] = bytearray()
        connection_data[conn_id]["last_processing_time"] = time.time()
        empty_transcription_counts[conn_id] = 0
        
        await websocket.send_text(json.dumps({
            "type": "listening_status",
            "message": "Listening for clear speech..."
        }))

async def handle_end_stream(websocket, payload: dict, conn_id: str):
    """Finalize live streaming session"""
    try:
        if not connection_data[conn_id]["streaming"]:
            return
            
        connection_data[conn_id]["streaming"] = False
        
        # Cancel any ongoing TTS
        if connection_data[conn_id].get("current_tts_task"):
            connection_data[conn_id]["current_tts_task"].cancel()
            connection_data[conn_id]["current_tts_task"] = None
        
        connection_data[conn_id]["maya_speaking"] = False
        
        await websocket.send_text(json.dumps({
            "type": "stream_ended",
            "message": "Live streaming session ended",
            "total_chunks_processed": connection_data[conn_id]["stream_chunk_count"]
        }))
        
        # Clean up tracking data
        if conn_id in response_tracker:
            del response_tracker[conn_id]
        if conn_id in empty_transcription_counts:
            del empty_transcription_counts[conn_id]
        
        print(f"🏁 Ended live streaming session for {conn_id}")
        
    except Exception as e:
        print(f"❌ Error ending stream for {conn_id}: {e}")

async def cleanup_on_error(conn_id: str):
    """Clean up connection state on error"""
    if conn_id in connection_data:
        connection_data[conn_id]["maya_speaking"] = False
        connection_data[conn_id]["stream_buffer"] = bytearray()

def cleanup_old_connections():
    """Clean up old connection data"""
    current_time = time.time()
    cutoff_time = current_time - 3600  # 1 hour
    
    # Clean up old response tracker data
    for conn_id in list(response_tracker.keys()):
        tracker = response_tracker[conn_id]
        # Remove old responses
        tracker["recent_responses"] = [
            (transcript, response, timestamp) 
            for transcript, response, timestamp in tracker["recent_responses"]
            if timestamp > cutoff_time
        ]
        
        # Remove empty trackers
        if not tracker["recent_responses"]:
            del response_tracker[conn_id]

# -------------------- CONVERSATION MANAGEMENT ENDPOINTS --------------------

@app.post("/reset_conversation/{user_id}")
async def reset_conversation(user_id: str):
    """Reset user's conversation history"""
    try:
        reset_user_conversation(user_id)
        return {
            "message": f"Conversation reset for user {user_id}",
            "status": "success"
        }
    except Exception as e:
        print(f"Error resetting conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset conversation")

@app.get("/conversation_status/{user_id}")
async def conversation_status(user_id: str):
    """Get conversation status for user"""
    try:
        conversation = get_user_conversation(user_id)
        return {
            "user_id": user_id,
            "message_count": len(conversation),
            "last_message": conversation[-1]["content"][:100] if conversation else "No messages",
            "status": "success"
        }
    except Exception as e:
        print(f"Error getting conversation status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get conversation status")

# -------------------- HEALTH CHECK --------------------

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper_model": "loaded",
        "audio_directory": "ready",
        "database": "ready",
        "maya_system": "ready",
        "active_conversations": len(user_conversations),
        "active_websockets": len(connection_data),
        "streaming_support": "enabled",
        "tts_support": "enabled"
    }

# Health check for WebSocket
@app.get("/ws/health")
async def websocket_health():
    """WebSocket health check"""
    return {
        "websocket_status": "healthy",
        "active_connections": len(connection_data),
        "streaming_enabled": True
    }

# -------------------- DEBUGGING ENDPOINTS --------------------

@app.get("/debug/connections")
async def debug_connections():
    """Debug endpoint to check active connections"""
    debug_info = {}
    for conn_id, data in connection_data.items():
        debug_info[conn_id] = {
            "user_id": data.get("user_id"),
            "streaming": data.get("streaming"),
            "buffer_size": len(data.get("stream_buffer", [])),
            "chunk_count": data.get("stream_chunk_count", 0)
        }
    return {"active_connections": debug_info}

@app.post("/debug/test_tts")
async def test_tts_endpoint(text: str = "Hello, this is a test message"):
    """Test TTS generation independently"""
    try:
        chunks = []
        async for chunk in text_to_speech_base64_chunks(text):
            if chunk:
                chunks.append(len(chunk))
        
        return {
            "status": "success",
            "text": text,
            "chunks_generated": len(chunks),
            "chunk_sizes": chunks
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# -------------------- STARTUP VALIDATION --------------------

@app.on_event("startup")
async def startup_event():
    """Validate system on startup"""
    print("🚀 Starting Maya WebSocket server...")
    
    # Test TTS
    try:
        test_chunks = []
        async for chunk in text_to_speech_base64_chunks("System startup test"):
            if chunk:
                test_chunks.append(chunk)
        print(f"✅ TTS system working - generated {len(test_chunks)} chunks")
    except Exception as e:
        print(f"❌ TTS system error: {e}")
    
    # Test Whisper
    try:
        # Create a small test audio file
        test_audio = np.random.rand(16000).astype(np.float32) * 0.1  # 1 second of quiet noise
        test_result = whisper_model.transcribe(test_audio)
        print("✅ Whisper model working")
    except Exception as e:
        print(f"❌ Whisper model error: {e}")
    
    print("🎉 Maya WebSocket server started successfully!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)