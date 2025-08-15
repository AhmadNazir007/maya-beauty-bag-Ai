# main.py - Updated with Maya system integration for voice
import os
import json
import uuid
import base64
import tempfile
import asyncio
import hashlib
from typing import List, Optional, Dict

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

# Initialize FastAPI app
app = FastAPI(title="Maya Beauty Bag AI API", version="1.0.0")

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
whisper_model = whisper.load_model("base")
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

🗣️ Voice Chat Note:
If the user is speaking (via voice), favor shorter, clearer, and more conversational responses. Imagine you're speaking aloud with softness and clarity. Avoid long paragraphs.

🧠 Voice Mode Context:
Do not reintroduce Maya at the beginning of every voice message. Assume the chat is continuous unless the user asks for a reset.

💡 If `mode == voice`, you can reduce verbosity and keep a warm tone.
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
    """Transcribe audio file to text using Whisper"""
    temp_file = None
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        if not file.content_type or 'audio' not in file.content_type:
            print(f"Warning: Unexpected content type: {file.content_type}")

        # Read file content
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        if len(content) < 100:  # Very small file
            raise HTTPException(status_code=400, detail="Audio file too small")

        # Create temp file with proper extension
        file_extension = 'webm'
        if file.content_type:
            if 'mp4' in file.content_type:
                file_extension = 'mp4'
            elif 'wav' in file.content_type:
                file_extension = 'wav'

        temp_file = f"temp_{uuid.uuid4().hex}.{file_extension}"

        # Save uploaded file
        with open(temp_file, "wb") as f:
            f.write(content)

        print(f"Processing audio file: {temp_file} ({len(content)} bytes)")

        # Transcribe with Whisper
        result = whisper_model.transcribe(
            temp_file,
            language='en',  # Force English for better results
            task='transcribe',
            fp16=False,  # More stable
            temperature=0.0,  # More deterministic
            best_of=1,  # Faster processing
            beam_size=1,  # Faster processing
            patience=1.0,
            condition_on_previous_text=False,
            initial_prompt="This is a conversation about beauty products and personal care."
        )

        transcribed_text = result["text"].strip()

        if not transcribed_text:
            raise HTTPException(status_code=400, detail="No speech detected in audio")

        # Filter out common Whisper hallucinations
        hallucinations = [
            "Thank you.", "Thanks for watching.", "Bye.", "you", "Thank you for watching.",
            "Please subscribe.", "Like and subscribe.", ".", "", " "
        ]

        if transcribed_text.lower().strip() in [h.lower() for h in hallucinations]:
            raise HTTPException(status_code=400, detail="No meaningful speech detected")

        print(f"Transcription successful: '{transcribed_text}'")

        return JSONResponse({
            "text": transcribed_text,
            "status": "success",
            "confidence": result.get("language_probability", 0.0) if "language_probability" in result else 1.0,
            "language": result.get("language", "en")
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        # Clean up temp file
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

async def transcribe_temp_file(path: str) -> str:
    """Transcribe a file path using Whisper model"""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
            
        file_size = os.path.getsize(path)
        if file_size == 0:
            raise ValueError(f"Audio file is empty: {path}")
            
        print(f"Transcribing file: {path} ({file_size} bytes)")
        
        result = whisper_model.transcribe(
            path,
            language='en',
            task='transcribe',
            fp16=False,
            temperature=0.0,
            best_of=1,
            beam_size=1,
            patience=1.0,
            condition_on_previous_text=False,
            initial_prompt="This is a conversation about beauty products and personal care."
        )
        
        transcript = result.get("text", "").strip()
        print(f"Transcription result: '{transcript}'")
        return transcript
        
    except Exception as e:
        print(f"Transcription error: {e}")
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

def text_to_speech_base64_chunks(text: str, chunk_size: int = 32 * 1024):
    """Generate base64-encoded audio chunks from text"""
    import re
    from gtts import gTTS
    
    try:
        clean_text = text.replace("*", "").replace("_", "").replace("#", "")
        clean_text = re.sub(r'[^\w\s.,!?;:-]', '', clean_text)
        if len(clean_text) > 1000:
            clean_text = clean_text[:1000] + "..."

        if not clean_text.strip():
            clean_text = "I'm sorry, I didn't catch that."

        # Ensure audio_responses directory exists
        os.makedirs("audio_responses", exist_ok=True)

        filename = f"maya_ws_{uuid.uuid4().hex[:8]}.mp3"
        out_path = os.path.join("audio_responses", filename)

        # Generate TTS
        tts = gTTS(text=clean_text, lang='en', slow=False, tld='com')
        tts.save(out_path)

        # Verify file was created
        if not os.path.exists(out_path):
            print(f"TTS file was not created: {out_path}")
            return
            
        file_size = os.path.getsize(out_path)
        if file_size == 0:
            print(f"TTS file is empty: {out_path}")
            return

        print(f"TTS file created: {out_path} ({file_size} bytes)")

        try:
            # Read and yield chunks
            with open(out_path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield base64.b64encode(chunk).decode('ascii')
        except Exception as read_error:
            print(f"Error reading TTS file {out_path}: {read_error}")
        finally:
            # Clean up the file
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
                    print(f"Cleaned up TTS file: {out_path}")
            except Exception as cleanup_error:
                print(f"Error cleaning up TTS file: {cleanup_error}")
                
    except Exception as e:
        print(f"TTS generation error: {e}")
        return

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("WebSocket connection request received")
    await websocket.accept()
    conn_id = str(uuid.uuid4())
    
    # Initialize connection data
    connection_data[conn_id] = {
        "audio_buffer": bytearray(),
        "user_id": "anonymous"
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

async def handle_audio_chunk(websocket: WebSocket, payload: dict, conn_id: str):
    """Handle incoming audio chunks"""
    b64 = payload.get("data")
    if not b64:
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": "Empty audio chunk"
        }))
        return
        
    try:
        chunk_bytes = base64.b64decode(b64)
        connection_data[conn_id]["audio_buffer"].extend(chunk_bytes)
        logger.debug(f"Added {len(chunk_bytes)} bytes to buffer for {conn_id}")
    except Exception as e:
        logger.error(f"Failed to decode audio chunk from {conn_id}: {e}")
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": "Invalid audio chunk encoding"
        }))

async def handle_end_audio(websocket: WebSocket, payload: dict, conn_id: str):
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
        
        print(f"Wrote {len(buf)} bytes to {tmp_path} for transcription")

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
            # Generate Maya's response using full system
            await websocket.send_text(json.dumps({
                "type": "status", 
                "message": "Maya is thinking..."
            }))
            
            maya_text = await generate_maya_response_websocket(transcript, user_id)
            await websocket.send_text(json.dumps({
                "type": "ai_text", 
                "data": maya_text
            }))

            # Stream TTS chunks
            await websocket.send_text(json.dumps({
                "type": "status", 
                "message": "Generating audio response..."
            }))
            
            chunk_count = 0
            try:
                for b64_chunk in text_to_speech_base64_chunks(maya_text):
                    if b64_chunk:  # Only send non-empty chunks
                        await websocket.send_text(json.dumps({
                            "type": "tts_chunk", 
                            "data": b64_chunk
                        }))
                        chunk_count += 1
                        await asyncio.sleep(0.01)  # Small delay

                print(f"Sent {chunk_count} TTS chunks to {conn_id}")
                
                # Notify TTS completion
                await websocket.send_text(json.dumps({
                    "type": "tts_end", 
                    "message": "Audio response complete"
                }))
            except Exception as tts_error:
                print(f"TTS streaming error: {tts_error}")
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": "Audio generation failed, but text response is available"
                }))
                
        else:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": "No speech detected in audio"
            }))

    except Exception as e:
        print(f"Error handling end_audio for {conn_id}: {e}")
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
                print(f"Cleaned up temp file: {tmp_path}")
            except Exception as cleanup_error:
                print(f"Error cleaning up temp file: {cleanup_error}")

async def handle_text_message(websocket: WebSocket, payload: dict, conn_id: str):
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
        logger.info(f"Processing text message from {user_id}: {user_text[:50]}...")
        
        # Send processing status
        await websocket.send_text(json.dumps({
            "type": "status", 
            "message": "Maya is processing your message..."
        }))

        # Generate Maya's response using full system
        maya_text = await generate_maya_response_websocket(user_text, user_id)
        await websocket.send_text(json.dumps({
            "type": "ai_text", 
            "data": maya_text
        }))

        # Send TTS status
        await websocket.send_text(json.dumps({
            "type": "status", 
            "message": "Generating audio response..."
        }))

        # Stream TTS chunks
        chunk_count = 0
        for b64_chunk in text_to_speech_base64_chunks(maya_text):
            await websocket.send_text(json.dumps({
                "type": "tts_chunk", 
                "data": b64_chunk
            }))
            chunk_count += 1
            await asyncio.sleep(0.01)

        logger.info(f"Sent {chunk_count} TTS chunks to {conn_id}")

        # Notify completion
        await websocket.send_text(json.dumps({
            "type": "tts_end", 
            "message": "Response complete"
        }))

    except Exception as e:
        logger.error(f"Error processing text message from {conn_id}: {e}")
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": f"Processing error: {str(e)}"
        }))

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
        "active_websockets": len(connection_data)
    }

# Health check for WebSocket
@app.get("/ws/health")
async def websocket_health():
    """Health check for WebSocket service"""
    return {
        "status": "healthy",
        "active_connections": len(connection_data),
        "websocket_endpoint": "/ws",
        "maya_integration": "enabled"
    }