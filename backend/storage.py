# backend/storage.py
import json
import os
from datetime import datetime

DATA_DIR = "users"

def save_user_order(user_id: str, order_data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{user_id}.json")

    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {"user_id": user_id, "orders": []}

    order_data["date"] = datetime.now().strftime("%Y-%m-%d")
    data["orders"].append(order_data)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def get_user_history(user_id: str):
    path = os.path.join(DATA_DIR, f"{user_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("orders", [])
