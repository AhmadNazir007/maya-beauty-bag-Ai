# backend/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, json
from .auth import hash_password, authenticate_user
from .storage import save_user_order, get_user_history

app = FastAPI()

USERS_DB_PATH = "users_db.json"

# --- Models ---
class UserRegister(BaseModel):
    username: str
    password: str

class OrderRequest(BaseModel):
    user_id: str
    bag: str
    products: list[str]
    affirmation: str

# --- Helpers ---
def load_users_db():
    if os.path.exists(USERS_DB_PATH):
        with open(USERS_DB_PATH, "r") as f:
            return json.load(f)
    return {}

def save_users_db(db):
    with open(USERS_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

# --- API ---
@app.post("/register")
def register(user: UserRegister):
    db = load_users_db()
    if user.username in db:
        raise HTTPException(status_code=400, detail="Username already exists")
    db[user.username] = {"password": hash_password(user.password)}
    save_users_db(db)
    return {"message": "User registered successfully"}

@app.post("/login")
def login(user: UserRegister):
    db = load_users_db()
    if authenticate_user(user.username, user.password, db):
        return {"message": "Login successful", "user_id": user.username}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/save_order")
def save_order(order: OrderRequest):
    save_user_order(order.user_id, {
        "bag": order.bag,
        "products": order.products,
        "affirmation": order.affirmation
    })
    return {"message": "Order saved successfully"}

@app.get("/order_history/{user_id}")
def order_history(user_id: str):
    orders = get_user_history(user_id)
    return {"orders": orders}
