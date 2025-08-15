import streamlit as st
from openai import OpenAI
import os
import json
import requests
from dotenv import load_dotenv
import time
import base64

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load product data
with open("data/products.json") as f:
    product_data = json.load(f)

bags = product_data["bags"]
products = product_data["products"]

API_LOCAL = "http://localhost:8000"
API_LIVE = "https://maya-beauty-bag-ai.onrender.com"
API = API_LIVE

# --- Utility Functions ---
def get_bag_options():
    return bags

def get_products_by_category(category):
    return products.get(category, [])

def create_checkout_summary(bag, selected_products, affirmation):
    return {
        "summary": f"You selected {bag} with {', '.join(selected_products)}. Your affirmation: {affirmation}."
    }

def save_order_to_backend(username, bag, products, affirmation, summary):
    try:
        res = requests.post(f"{API}/save_order", json={
            "user_id": username,
            "bag": bag,
            "products": products,
            "affirmation": affirmation,
            "summary": summary
        })
        if res.status_code == 200:
            st.success("✅ Order saved to backend!")
        else:
            st.warning("⚠️ Could not save order.")
    except Exception as e:
        st.error(f"❌ Error saving order: {e}")

def fetch_order_history(username):
    try:
        res = requests.get(f"{API}/order_history/{username}")
        if res.status_code == 200:
            return res.json().get("orders", [])
    except:
        pass
    return []

def check_server_status():
    """Check if the backend server is available"""
    try:
        response = requests.get(f"{API}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def process_voice_input(transcribed_text):
    """Process transcribed voice input through Maya's conversation system"""
    if not transcribed_text or transcribed_text.strip() == "":
        return
        
    transcribed_text = transcribed_text.strip()
    
    # Add user message to conversation
    st.session_state.messages.append({"role": "user", "content": transcribed_text})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(f"🎤 *{transcribed_text}*")

    # Process through Maya's system using the new backend endpoint
    with st.chat_message("assistant"):
        with st.spinner("Maya is thinking..."):
            # Use the new Maya-integrated endpoint for voice processing
            if check_server_status():
                try:
                    # Call the new Maya voice processing endpoint
                    maya_response = requests.post(f"{API}/process_voice_with_maya", 
                        json={
                            "message": transcribed_text,
                            "user_id": st.session_state.username,
                            "reset_conversation": False
                        }, 
                        timeout=15)
                    
                    if maya_response.status_code == 200:
                        response_data = maya_response.json()
                        response = response_data.get("text_response", "I'm sorry, I couldn't process that.")
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Generate audio response
                        try:
                            audio_response = requests.post(f"{API}/generate_audio", 
                                json={
                                    "text": response, 
                                    "user_id": st.session_state.username
                                }, 
                                timeout=10)
                                
                            if audio_response.status_code == 200 and "audio_path" in audio_response.json():
                                audio_url = f"{API}{audio_response.json()['audio_path']}"
                                st.audio(audio_url, autoplay=True)
                                st.success("🎵 Audio generated successfully!")
                            else:
                                st.info("💬 Text response ready (audio unavailable)")
                                
                        except requests.exceptions.Timeout:
                            st.info("💬 Text response ready (audio timed out)")
                        except Exception as e:
                            st.info(f"💬 Text response ready (audio error: {str(e)})")
                    else:
                        st.error(f"Error: {maya_response.json().get('detail', 'Processing failed')}")
                        
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to server.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                # Fallback to local processing if server is offline
                response = get_maya_response(st.session_state.messages)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.info("💬 Text response ready (server offline)")

def reset_conversation():
    """Reset the conversation history"""
    if check_server_status():
        try:
            # Reset on server
            response = requests.post(f"{API}/reset_conversation/{st.session_state.username}", timeout=5)
            if response.status_code == 200:
                st.success("✅ Server conversation reset!")
        except Exception as e:
            st.warning(f"⚠️ Could not reset server conversation: {e}")

# --- Tool Definitions ---
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

# --- System Prompt ---
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


# Reset local conversation
st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "assistant", 
            "content": "Hey love, I'm Maya. This isn't just beauty — it's a moment just for you. You ready to create something that celebrates your vibe, your softness, your fire? Let's start with your bag — your beauty ritual begins there."
        }
    ]
st.success("🔄 Local conversation reset!")

# --- LLM Chat Logic ---
def get_maya_response(convo):
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=convo,
        tools=tool_definitions,
        tool_choice="auto",
        temperature=0.8
    )
    message = response.choices[0].message

    if hasattr(message, "tool_calls") and message.tool_calls:
        tool_call = message.tool_calls[0]
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if func_name == "get_bag_options":
            result = get_bag_options()
        elif func_name == "get_products_by_category":
            result = get_products_by_category(args["category"])
        elif func_name == "create_checkout_summary":
            result = create_checkout_summary(
                args["bag"], args["selected_products"], args["affirmation"]
            )
            save_order_to_backend(
                st.session_state.username,
                args["bag"], args["selected_products"], args["affirmation"],
                result["summary"]
            )

        convo.append({"role": "assistant", "tool_calls": [tool_call.model_dump()]})
        convo.append({
            "role": "tool", 
            "tool_call_id": tool_call.id, 
            "name": func_name, 
            "content": ""
        })
        return get_maya_response(convo)

    return message.content.strip()

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Maya - Beauty in a Bag", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "username" not in st.session_state:
    st.session_state.username = None

if "voice_input_received" not in st.session_state:
    st.session_state.voice_input_received = False

if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

if "recording_status" not in st.session_state:
    st.session_state.recording_status = "ready"

if "processing_voice" not in st.session_state:
    st.session_state.processing_voice = False

if "server_status" not in st.session_state:
    st.session_state.server_status = check_server_status()

# --- LOGIN/REGISTRATION SYSTEM ---
if st.session_state.username is None:
    st.title("✨ Welcome to Maya – Beauty in a Bag")
    st.markdown("*Your personal beauty ritual awaits*")
    
    # Show server status
    if not st.session_state.server_status:
        st.warning("⚠️ Backend server is offline. Some features may be limited.")
        if st.button("🔄 Check Server Status"):
            st.session_state.server_status = check_server_status()
            st.rerun()
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

    with tab1:
        st.subheader("Sign In")
        login_user = st.text_input("Username", key="login_user", placeholder="Enter your username")
        login_pass = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Login", use_container_width=True):
                if login_user and login_pass:
                    if st.session_state.server_status:
                        try:
                            res = requests.post(f"{API}/login", json={
                                "username": login_user, 
                                "password": login_pass
                            }, timeout=10)
                            if res.status_code == 200:
                                st.session_state.username = login_user
                                st.success("✅ Login successful!")
                                st.rerun()
                            else:
                                st.error(res.json().get("detail", "Login failed"))
                        except requests.exceptions.ConnectionError:
                            st.error("❌ Cannot connect to server. Please ensure the FastAPI server is running.")
                            st.session_state.server_status = False
                        except requests.exceptions.Timeout:
                            st.error("❌ Request timed out. Please try again.")
                        except Exception as e:
                            st.error(f"❌ Login error: {e}")
                    else:
                        # Demo mode login
                        if login_user and login_pass:
                            st.session_state.username = login_user
                            st.info("✅ Logged in (Demo Mode - Server Offline)")
                            st.rerun()
                else:
                    st.warning("Please fill in both username and password")

    with tab2:
        st.subheader("Create Account")
        reg_user = st.text_input("New Username", key="reg_user", placeholder="Choose a username")
        reg_pass = st.text_input("New Password", type="password", key="reg_pass", placeholder="Create a password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎉 Register", use_container_width=True):
                if reg_user and reg_pass:
                    if st.session_state.server_status:
                        try:
                            res = requests.post(f"{API}/register", json={
                                "username": reg_user, 
                                "password": reg_pass
                            }, timeout=10)
                            if res.status_code == 200:
                                st.success("🎉 Registered successfully! You can now log in.")
                            else:
                                st.error(res.json().get("detail", "Registration failed"))
                        except requests.exceptions.ConnectionError:
                            st.error("❌ Cannot connect to server. Please ensure the FastAPI server is running.")
                            st.session_state.server_status = False
                        except requests.exceptions.Timeout:
                            st.error("❌ Request timed out. Please try again.")
                        except Exception as e:
                            st.error(f"❌ Registration error: {e}")
                    else:
                        st.info("🎉 Registration noted (Demo Mode - Server Offline)")
                else:
                    st.warning("Please fill in both username and password")
    
    st.stop()

# --- SIDEBAR --- (Replace the existing sidebar section)
with st.sidebar:
    st.title("Maya 💄")
    st.write(f"👤 **{st.session_state.username}**")
    
    # Server status indicator
    if st.session_state.server_status:
        st.success("🟢 Server Online")
    else:
        st.error("🔴 Server Offline")
        if st.button("🔄 Refresh Status"):
            st.session_state.server_status = check_server_status()
            st.rerun()
    
    st.divider()
    
    # Chat mode selection
    chat_mode = st.radio(
        "💬 Chat Mode", 
        ["Text", "Voice"], 
        key="mode_select",
        help="Choose how you want to interact with Maya"
    )
    
    if chat_mode == "Voice" and not st.session_state.server_status:
        st.warning("⚠️ Voice mode requires server connection")
    
    st.divider()
    
    # Conversation management
    st.subheader("💭 Conversation")
    
    if st.button("🔄 Reset Conversation"):
        reset_conversation()
        st.rerun()
    
    # Show conversation stats
    if st.session_state.server_status:
        try:
            conv_status = requests.get(f"{API}/conversation_status/{st.session_state.username}", timeout=5)
            if conv_status.status_code == 200:
                data = conv_status.json()
                st.write(f"📊 Messages: {data.get('message_count', 0)}")
        except:
            pass
    
    local_msg_count = len([m for m in st.session_state.messages if m.get("role") in ["user", "assistant"]])
    st.write(f"🏠 Local: {local_msg_count} messages")
    
    st.divider()
    
    # Order history
    if st.button("📦 View Order History"):
        if st.session_state.server_status:
            orders = fetch_order_history(st.session_state.username)
            if orders:
                st.write("**Your Orders:**")
                for i, order in enumerate(orders[-3:], 1):  # Show last 3 orders
                    st.write(f"{i}. {order.get('summary', 'Order placed')}")
            else:
                st.write("No orders yet!")
        else:
            st.info("Order history unavailable (server offline)")
    
    st.divider()
    
      # Logout
    if st.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- INITIALIZE CONVERSATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "assistant", 
            "content": "Hey love, I'm Maya. This isn't just beauty — it's a moment just for you. You ready to create something that celebrates your vibe, your softness, your fire? Let's start with your bag — your beauty ritual begins there."
        }
    ]

# --- HANDLE VOICE INPUT FROM URL PARAMETERS ---
query_params = st.query_params
if "voice_input" in query_params and not st.session_state.processing_voice:
    transcribed_text = query_params.get("voice_input", "").strip()
    if transcribed_text:
        st.session_state.processing_voice = True
        
        # Process the voice input
        st.session_state.messages.append({"role": "user", "content": transcribed_text})
        
        # Get Maya's response
        with st.spinner("Maya is responding..."):
            response = get_maya_response(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Generate audio response only if server is available
            if st.session_state.server_status:
                try:
                    audio_response = requests.post(
                        f"{API}/generate_audio", 
                        json={
                            "text": response, 
                            "user_id": st.session_state.username
                        }, 
                        timeout=10
                    )
                        
                    if audio_response.status_code == 200 and "audio_path" in audio_response.json():
                        audio_url = f"{API}{audio_response.json()['audio_path']}"
                        st.session_state.pending_audio = audio_url
                        
                except Exception as e:
                    st.session_state.audio_error = str(e)
        
        # Clear the query param and reset processing flag
        st.query_params.clear()
        st.session_state.processing_voice = False
        st.rerun()

# --- DISPLAY CONVERSATION HISTORY ---
st.title("✨ Maya - Your Beauty Assistant")

# Display conversation
for msg in st.session_state.messages[1:]:  # Skip system message
    if msg["role"] in {"tool", "function"} or "content" not in msg:
        continue
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and st.session_state.mode_select == "Voice":
            st.markdown(f"🎤 *{msg['content']}*")
        else:
            st.markdown(msg["content"])

# Play pending audio if available
if "pending_audio" in st.session_state:
    st.audio(st.session_state.pending_audio, autoplay=True)
    st.success("🎵 Audio generated successfully!")
    del st.session_state.pending_audio

# Show audio error if any
if "audio_error" in st.session_state:
    st.warning(f"⚠️ Audio generation failed: {st.session_state.audio_error}")
    del st.session_state.audio_error

# --- MAIN CHAT INTERFACE ---
if st.session_state.mode_select == "Text":
    # TEXT MODE
    if prompt := st.chat_input("Your reply...", key="text_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Maya is thinking..."):
                response = get_maya_response(st.session_state.messages)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

elif st.session_state.mode_select == "Voice":
    # VOICE MODE
    if not st.session_state.server_status:
        st.error("🔴 Voice mode is currently unavailable because the backend server is offline.")
        st.info("💡 Please start your FastAPI server or use Text mode instead.")
    else:
        st.subheader("🎤 Speak to Maya")
        
        # Enhanced voice interface information
        st.info("🎯 **Maya's Voice Mode Features:**\n- Full tool access for bag selection and product recommendations\n- Conversation memory across sessions\n- Order placement and history\n- Personalized affirmations")
        
        # Display voice interface
        st.components.v1.html(open('voice_interface.html', encoding='utf-8').read(), height=300)

        # Alternative text input for voice mode
        st.divider()
        st.markdown("**Alternative: Type your message to Maya**")
        
        voice_input_key = f"voice_text_input_{int(time.time())}"
        voice_send_key = f"voice_text_send_{int(time.time())}"
        
        col1, col2 = st.columns([4, 1])
        with col1:
            voice_text = st.text_input("Or type what you want to say to Maya:", key=voice_input_key)
        with col2:
            if st.button("Send", key=voice_send_key, use_container_width=True):
                if voice_text and voice_text.strip():
                    # Process using Maya's full system
                    st.session_state.messages.append({"role": "user", "content": voice_text})
                    with st.chat_message("user"):
                        st.markdown(f"💬 {voice_text}")

                    with st.chat_message("assistant"):
                        with st.spinner("Maya is thinking..."):
                            # Use Maya-integrated endpoint
                            try:
                                maya_response = requests.post(f"{API}/process_voice_with_maya", 
                                    json={
                                        "message": voice_text,
                                        "user_id": st.session_state.username,
                                        "reset_conversation": False
                                    }, 
                                    timeout=15)
                                
                                if maya_response.status_code == 200:
                                    response_data = maya_response.json()
                                    response = response_data.get("text_response", "I'm sorry, I couldn't process that.")
                                    
                                    st.markdown(response)
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                                    
                                    # Generate audio response
                                    try:
                                        audio_response = requests.post(
                                            f"{API}/generate_audio", 
                                            json={
                                                "text": response, 
                                                "user_id": st.session_state.username
                                            }, 
                                            timeout=10
                                        )
                                            
                                        if audio_response.status_code == 200 and "audio_path" in audio_response.json():
                                            audio_url = f"{API}{audio_response.json()['audio_path']}"
                                            st.audio(audio_url, autoplay=True)
                                            st.success("🎵 Audio generated successfully!")
                                        else:
                                            st.info("💬 Text response ready (audio unavailable)")
                                            
                                    except requests.exceptions.Timeout:
                                        st.info("💬 Text response ready (audio timed out)")
                                    except Exception as e:
                                        st.info(f"💬 Text response ready (audio error: {str(e)})")
                                else:
                                    st.error(f"Error: {maya_response.json().get('detail', 'Processing failed')}")
                                    
                            except Exception as e:
                                st.error(f"❌ Error processing message: {str(e)}")
                    
                    st.rerun()

 # Show Maya's capabilities in voice mode
        with st.expander("🔧 Maya's Capabilities", expanded=False):
            st.markdown("""
            **What Maya can do in Voice Mode:**
            - 🛍️ **Bag Selection**: "Show me bag options" or "I want a bold bag"
            - 🎨 **Product Recommendations**: "What products do you have for eyes?"
            - 💝 **Order Creation**: "I want the Glow Ritual with mascara and lipstick"
            - 💎 **Affirmations**: Maya creates personalized affirmations based on your choices
            - 📝 **Order History**: "Show me my previous orders"
            - 💬 **Beauty Advice**: Ask about skincare, makeup tips, or beauty routines
            """)

# Show recent conversation in voice mode
    if len(st.session_state.messages) > 2:
        with st.expander("💬 Recent Conversation", expanded=False):
            recent_messages = st.session_state.messages[-6:]
            for msg in recent_messages:
                content = msg.get("content")
                if not content:
                    continue  # Skip messages without 'content'
                
                if msg.get("role") == "user":
                    st.write(f"**You:** {content}")
                elif msg.get("role") == "assistant":
                    st.write(f"**Maya:** {content[:100]}{'...' if len(content) > 100 else ''}")

# Show Maya system status
if st.session_state.server_status:
    with st.expander("🔍 Maya System Status", expanded=False):
        try:
            health_response = requests.get(f"{API}/health", timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                st.json(health_data)
        except:
            st.write("Could not fetch system status")
            
# --- FOOTER ---
st.divider()
st.markdown("*✨ Powered by Maya AI - Your Beauty Ritual Companion*")

# Initialize voice component ID in session state for uniqueness
if "voice_component_id" not in st.session_state:
    st.session_state.voice_component_id = str(int(time.time()))