"""
main.py -- FastAPI server for JustiFi.
Exposes 4 endpoints that the React frontend calls:
  POST /express-draft
  POST /clause-checker
  POST /case-miner
  POST /legal-mind

Run:  uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import traceback
import os
import bcrypt
from pymongo import MongoClient
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import (
    query_express_draft,
    query_clause_checker,
    query_case_miner,
    query_legal_mind,
)
from file_utils import extract_text_from_file

# -- App Setup --------------------------------------------------------
app = FastAPI(
    title="JustiFi API",
    description="AI-Powered Indian Legal Research & Automation",
    version="1.0.0",
)

# CORS -- allow the React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Database Setup ---------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI")
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
use_mongo = False
use_postgres = False
users_collection = None

if NEON_DATABASE_URL:
    try:
        pg_conn = psycopg2.connect(NEON_DATABASE_URL)
        with pg_conn.cursor() as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL
                )
            ''')
        pg_conn.commit()
        pg_conn.close()
        use_postgres = True
        print("Connected to Neon PostgreSQL successfully.")
    except Exception as e:
        print(f"Neon PostgreSQL connection failed: {e}. Falling back to MongoDB/SQLite.")

if not use_postgres and MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        mongo_client.server_info() # Trigger connection test
        db = mongo_client["Justifi"]
        users_collection = db["users"]
        use_mongo = True
        print("Connected to MongoDB successfully.")
    except Exception as e:
        print(f"MongoDB connection failed: {e}. Falling back to local SQLite.")

# Fallback to SQLite if both Neon and MongoDB fail
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

if not use_postgres and not use_mongo:
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# -- Health Check -----------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok", "message": "JustiFi API is running"}


# =====================================================================
# Authentication Endpoints
# =====================================================================

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/signup")
async def signup(req: SignupRequest):
    hashed_password = bcrypt.hashpw(req.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    if use_postgres:
        try:
            pg_conn = psycopg2.connect(NEON_DATABASE_URL)
            with pg_conn.cursor() as cur:
                cur.execute('INSERT INTO users (name, email, password) VALUES (%s, %s, %s)',
                             (req.name, req.email, hashed_password))
            pg_conn.commit()
            pg_conn.close()
        except psycopg2.IntegrityError:
            raise HTTPException(status_code=400, detail="User with this email already exists")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    elif use_mongo:
        if users_collection.find_one({"email": req.email}):
            raise HTTPException(status_code=400, detail="User with this email already exists")
        
        users_collection.insert_one({
            "name": req.name,
            "email": req.email,
            "password": hashed_password
        })
    else:
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                         (req.name, req.email, hashed_password))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            raise HTTPException(status_code=400, detail="User with this email already exists")
        conn.close()
        
    return {"message": "Signup successful"}

@app.post("/login")
async def login(req: LoginRequest):
    if use_postgres:
        pg_conn = psycopg2.connect(NEON_DATABASE_URL)
        with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM users WHERE email = %s', (req.email,))
            user = cur.fetchone()
        pg_conn.close()
        
        if not user:
            raise HTTPException(status_code=400, detail="Invalid email or password")
            
        if not bcrypt.checkpw(req.password.encode('utf-8'), user["password"].encode('utf-8')):
            raise HTTPException(status_code=400, detail="Invalid email or password")
            
        name = user["name"]
    elif use_mongo:
        user = users_collection.find_one({"email": req.email})
        if not user:
            raise HTTPException(status_code=400, detail="Invalid email or password")
        
        if not bcrypt.checkpw(req.password.encode('utf-8'), user["password"].encode('utf-8')):
            raise HTTPException(status_code=400, detail="Invalid email or password")
            
        name = user["name"]
    else:
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (req.email,)).fetchone()
        conn.close()
        
        if not user:
            raise HTTPException(status_code=400, detail="Invalid email or password")
            
        if not bcrypt.checkpw(req.password.encode('utf-8'), user["password"].encode('utf-8')):
            raise HTTPException(status_code=400, detail="Invalid email or password")
            
        name = user["name"]
        
    return {"message": "Login successful", "name": name}

@app.post("/reset-password")
async def reset_password(req: LoginRequest):
    hashed_password = bcrypt.hashpw(req.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    if use_postgres:
        pg_conn = psycopg2.connect(NEON_DATABASE_URL)
        with pg_conn.cursor() as cur:
            cur.execute('UPDATE users SET password = %s WHERE email = %s', (hashed_password, req.email))
            updated_rows = cur.rowcount
        pg_conn.commit()
        pg_conn.close()
        
        if updated_rows == 0:
            raise HTTPException(status_code=404, detail="User with this email not found")
    elif use_mongo:
        result = users_collection.update_one({"email": req.email}, {"$set": {"password": hashed_password}})
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User with this email not found")
    else:
        conn = get_db_connection()
        conn.execute('UPDATE users SET password = ? WHERE email = ?', (hashed_password, req.email))
        if conn.total_changes == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="User with this email not found")
        conn.commit()
        conn.close()
        
    return {"message": "Password reset successful"}


# =====================================================================
# Endpoint 1: Express Draft
# =====================================================================
@app.post("/express-draft")
async def express_draft(
    doc_type: str = Form("nda"),
    details: str = Form("{}"),
    file: Optional[UploadFile] = File(None)
):
    """Generate a legal document."""
    try:
        details_dict = json.loads(details)
        extra_text = ""
        if file and file.filename:
            file_bytes = await file.read()
            extra_text = extract_text_from_file(file_bytes, file.filename)
            
        result = query_express_draft(doc_type, details_dict, extra_text)
        return result
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "document": f"Error generating document: {e}", "sources": []}


# =====================================================================
# Endpoint 2: Clause Checker
# =====================================================================
@app.post("/clause-checker")
async def clause_checker(
    clause_text: str = Form(""),
    file: Optional[UploadFile] = File(None)
):
    """Analyse a legal clause."""
    try:
        extra_text = ""
        if file and file.filename:
            file_bytes = await file.read()
            extra_text = extract_text_from_file(file_bytes, file.filename)
            
        if not clause_text and not extra_text:
            return {"error": "No clause text or file provided", "answer": "", "risk_level": "", "sources": []}
            
        result = query_clause_checker(clause_text, extra_text)
        return result
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "answer": f"Error: {e}", "risk_level": "Unknown", "sources": []}


# =====================================================================
# Endpoint 3: Case Miner
# =====================================================================
@app.post("/case-miner")
async def case_miner(
    query: str = Form(""),
    file: Optional[UploadFile] = File(None)
):
    """Search for relevant legal cases."""
    try:
        extra_text = ""
        if file and file.filename:
            file_bytes = await file.read()
            extra_text = extract_text_from_file(file_bytes, file.filename)
            
        if not query and not extra_text:
            return {"error": "No query or file provided", "answer": "", "sources": []}
            
        result = query_case_miner(query, extra_text)
        return result
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "answer": f"Error: {e}", "sources": []}


# =====================================================================
# Endpoint 4: Legal Mind
# =====================================================================
@app.post("/legal-mind")
async def legal_mind(
    question: str = Form(""),
    context: str = Form(""),
    file: Optional[UploadFile] = File(None)
):
    """Deep legal reasoning and analysis."""
    try:
        extra_text = ""
        if file and file.filename:
            file_bytes = await file.read()
            extra_text = extract_text_from_file(file_bytes, file.filename)
            
        if not question and not extra_text:
            return {"error": "No question or file provided", "answer": "", "sources": []}
            
        result = query_legal_mind(question, context, extra_text)
        return result
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "answer": f"Error: {e}", "sources": []}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
