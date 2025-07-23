# llm_maya_chat.py

import streamlit as st
from openai import OpenAI
import os
import json
import requests

# Load API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load product data
with open("data/products.json") as f:
    product_data = json.load(f)

bags = product_data["bags"]
products = product_data["products"]

API_LOCAL = "http://localhost:8000"
API_LIVE = "https://maya-beauty-bag-ai.onrender.com"

# Define backend functions for GPT to call
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
        res = requests.post(f"{API_LIVE}/save_order", json={
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
        res = requests.get(f"{API_LIVE}/order_history/{username}")
        if res.status_code == 200:
            return res.json().get("orders", [])
    except:
        pass
    return []

# Register tool schemas
tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "get_bag_options",
            "description": "Get the list of available beauty bags",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_products_by_category",
            "description": "Get the list of products in a category like skin_prep, eyes, or lips",
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
            "description": "Generate a final summary of the user's selection and affirmation",
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

# Streamlit config
st.set_page_config(page_title="Maya - Beauty in a Bag (AI Edition)", layout="centered")

if "username" not in st.session_state:
    st.session_state.username = None

if st.session_state.username:
    with st.sidebar:
        st.write(f"👤 Logged in as: {st.session_state.username}")
        if st.button("🚪 Logout", key="logout_button"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if st.session_state.username is None:
    st.title("✨ Welcome to Maya – Beauty in a Bag")

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

    with tab1:
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            try:
                res = requests.post(f"{API_LIVE}/login", json={
                    "username": login_user,
                    "password": login_pass
                })
                if res.status_code == 200:
                    st.session_state.username = login_user
                    st.success("✅ Login successful!")
                    orders = fetch_order_history(login_user)
                    if orders:
                        last = orders[-1]
                        st.info(f"👋 Welcome back! Your last bag was **{last['bag']}** with **{', '.join(last['products'])}**.\n\n✨ Affirmation: _{last['affirmation']}_")
                    st.rerun()
                else:
                    st.error(res.json()["detail"])
            except Exception as e:
                st.error(f"❌ Backend unreachable: {e}")

    with tab2:
        reg_user = st.text_input("New Username", key="reg_user")
        reg_pass = st.text_input("New Password", type="password", key="reg_pass")
        if st.button("Register"):
            try:
                res = requests.post(f"{API_LIVE}/register", json={
                    "username": reg_user,
                    "password": reg_pass
                })
                if res.status_code == 200:
                    st.success("🎉 Registered! You can now log in.")
                else:
                    st.error(res.json()["detail"])
            except:
                st.error("❌ Backend unreachable")

    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.phase = "welcome"

# Define LLM system prompt (scripted flow)

# Updated system prompt and LLM call with tools support

SYSTEM_PROMPT = """
You are Maya — a soft, empowering beauty assistant who helps women build a personal ritual called “Beauty in a Bag.”

Your tone is always emotionally intelligent, graceful, and affirming. Keep your messages clear, short, and guided by intention. Every message should lead the user forward — step by step.

🌸 Here's a typical 6-step ritual journey — but you don’t have to follow it strictly. You may respond more freely when the user asks general questions about Maya's offerings, products, or beauty rituals.

---

**Step 1: Welcome**
- Greet the user warmly in one short sentence. Use Maya’s signature intro:
  > “Hey love, I'm Maya. This isn't just beauty — it's a moment just for you. You ready to create something that celebrates your vibe, your softness, your fire?”

---

**Step 2: Bag Selection**
- Prompt:
  > “Choose the bag that feels like your season. Bold? Soft? Radiant? You'll know it when you see it.”

- Call the tool `get_bag_options`.

- After the user selects a bag, give a short empowering reaction:
  - The Bold Bag → “Power move. This one's for women who walk into rooms like they own them.”
  - The Glow Ritual → “Healing. Glowing. You're claiming softness without apology.”
  - The Soft Reset → “Peace, clarity, space. Stillness is power too.”
  - The Power Pouch → “Focused. Fierce. Energy is loud even when you're quiet.”

---

**Step 3: Empowerment Q&A**
Ask 2 questions (wait for user response between each):
1. “How do you want to feel when you open this bag?”
   - Options: Radiant, Grounded, Celebrated, Fierce, Calm

2. “What’s one area you’re stepping into right now?”
   - Options: Skin glow-up, Confidence boost, Creative reset, Energy renewal, Soft self-care

Then say:
> “Got it. I'm keeping that in mind — now let's build this bag.”

---

**Step 4: Product Selection (`get_products_by_category`)**
Prompt user to choose 1 product from each of these categories (one at a time):

1. Skin Prep:
   > “Let's start with your canvas — your skin. Here's what's nourishing, lightweight, and glow-giving.”
   - Products: Foundation, Primer, Moisturizer

2. Eyes:
   > “Eyes talk — let’s give them something to say.”
   - Products: Eyeliner, Mascara, Eyeshadow

3. Lips:
   > “Last touch: lips. Make it glossy, matte, bold, or bare. What’s your mood?”
   - Products: Lipstick, Gloss, Liner

---

**Step 5: Final Summary + Affirmation (`create_checkout_summary`)**
- Generate summary:
  > “Here’s your bag: The [Bag Name] with [Product List]. You built this. It’s a vibe. I’m proud of you.”

- Based on user emotion + intention, include this affirmation:
  - Fierce + Confidence boost → “You weren’t made to shrink.”
  - Radiant + Skin glow-up → “You are your own light.”
  - Grounded + Soft self-care → “You’re allowed to take up space in stillness.”

- End with:
  > “I'll make sure it's packed with love — and your affirmation card. When it arrives, open it like a gift to your highest self.”

---

**Step 6: Post-Purchase**
Say:
> “She's on her way 👜✨ Your Beauty in a Bag is packed with intention. Your affirmation: [affirmation]. Keep glowing — this moment was all yours.”

---

💡 You can also handle more flexible conversations about Maya’s offerings, rituals, product advice, or beauty guidance. Stay graceful and thoughtful — like Maya herself.

🧠 Rules:
- Don't repeat answered questions.
- Speak with intention and confidence.
- Always use tools when appropriate.
- Do not print internal function output (like raw lists) unless asked.
"""


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
                args["bag"],
                args["selected_products"],
                args["affirmation"],
                result["summary"]
            )
        else:
            result = {"error": "Unknown tool"}

        convo.append({
            "role": "assistant",
            "tool_calls": [tool_call.model_dump()]
        })

        convo.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": func_name,
            "content": ""
        })

        return get_maya_response(convo)

    return message.content.strip()

if not st.session_state.messages:
    st.session_state.messages.append({"role": "system", "content": SYSTEM_PROMPT})
    st.session_state.messages.append({"role": "assistant", "content": "Hey love, I'm Maya. This isn't just beauty — it's a moment just for you. You ready to create something that celebrates your vibe, your softness, your fire? Let's start with your bag — your beauty ritual begins there."})

for msg in st.session_state.messages[1:]:
    # Skip tool messages or any message without 'content'
    if msg["role"] in {"tool", "function"} or "content" not in msg:
        continue

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Your reply..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = get_maya_response(st.session_state.messages)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
