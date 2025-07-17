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
        res = requests.post("http://localhost:8000/save_order", json={
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
        res = requests.get(f"http://localhost:8000/order_history/{username}")
        if res.status_code == 200:
            return res.json().get("orders", [])
    except:
        pass
    return []

# Register function schemas
function_definitions = [
    {
        "name": "get_bag_options",
        "description": "Get the list of available beauty bags",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_products_by_category",
        "description": "Get the list of products in a category like skin_prep, eyes, or lips",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string"}
            },
            "required": ["category"]
        }
    },
    {
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
]

# Streamlit config
st.set_page_config(page_title="Maya - Beauty in a Bag (AI Edition)", layout="centered")


# Init session state for login
if "username" not in st.session_state:
    st.session_state.username = None

# Logout button
if st.session_state.username:
    with st.sidebar:
        st.write(f"👤 Logged in as: {st.session_state.username}")
        if st.button("🚪 Logout", key="logout_button"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Display login/register form before chat
if st.session_state.username is None:
    st.title("✨ Welcome to Maya – Beauty in a Bag")

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

    with tab1:
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            try:
                res = requests.post("http://localhost:8000/login", json={
                    "username": login_user,
                    "password": login_pass
                })
                if res.status_code == 200:
                    st.session_state.username = login_user
                    st.success("✅ Login successful!")

                    # Show last order
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
                res = requests.post("http://localhost:8000/register", json={
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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.phase = "welcome"

# Define LLM system prompt (scripted flow)
SYSTEM_PROMPT = """
You are Maya, a warm, empowering beauty assistant that guides women in building a personal beauty ritual. Speak in a soft, affirming, emotionally intelligent tone.

Follow this exact 6-step conversational journey with the user:
... [truncated: full prompt from previous version remains here] ...
"""

# LLM call with tool (function) support
def get_maya_response(convo):
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=convo,
        temperature=0.8,
        functions=function_definitions,
        function_call="auto"
    )

    message = response.choices[0].message

    if message.function_call:
        func_name = message.function_call.name
        args = json.loads(message.function_call.arguments)

        if func_name == "get_bag_options":
            result = get_bag_options()
        elif func_name == "get_products_by_category":
            result = get_products_by_category(args["category"])
        elif func_name == "create_checkout_summary":
            result = create_checkout_summary(
                args["bag"], args["selected_products"], args["affirmation"]
            )
            # Save final order
            save_order_to_backend(
                st.session_state.username,
                args["bag"],
                args["selected_products"],
                args["affirmation"],
                result["summary"]
            )
        else:
            result = {"error": "Function not found"}

        convo.append({
            "role": "function",
            "name": func_name,
            "content": json.dumps(result)
        })

        return get_maya_response(convo)

    return message.content.strip()

# Start conversation
if not st.session_state.messages:
    st.session_state.messages.append({"role": "system", "content": SYSTEM_PROMPT})
    st.session_state.messages.append({"role": "assistant", "content": "Hey love, I'm Maya. This isn't just beauty — it's a moment just for you. You ready to create something that celebrates your vibe, your softness, your fire? Let's start with your bag — your beauty ritual begins there."})

# Display chat messages
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Your reply..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = get_maya_response(st.session_state.messages)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
