# backend/auth.py
import hashlib

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username: str, password: str, users_db: dict) -> bool:
    if username not in users_db:
        return False
    return users_db[username]["password"] == hash_password(password)
