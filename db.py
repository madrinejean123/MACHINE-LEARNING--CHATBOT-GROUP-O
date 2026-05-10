"""
db.py — SQLite storage for Safeguarding Companion
Includes simple email + password login + chat isolation by session_id
"""

import sqlite3
import hashlib
import uuid

DB_NAME = "chat.db"


# ==========================================================
# 1. INIT DATABASE
# ==========================================================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # USERS TABLE (LOGIN SYSTEM)
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        session_id TEXT UNIQUE NOT NULL
    )
    """)

    # MESSAGES TABLE (YOUR EXISTING ONE)
    c.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        message TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


# ==========================================================
# 2. HASH PASSWORD
# ==========================================================
def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()


# ==========================================================
# 3. REGISTER USER
# ==========================================================
def register_user(email: str, password: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    session_id = str(uuid.uuid4())

    try:
        c.execute("""
        INSERT INTO users (email, password, session_id)
        VALUES (?, ?, ?)
        """, (email, hash_password(password), session_id))

        conn.commit()
        return session_id

    except sqlite3.IntegrityError:
        return None

    finally:
        conn.close()


# ==========================================================
# 4. LOGIN USER
# ==========================================================
def login_user(email: str, password: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    SELECT session_id, password
    FROM users
    WHERE email = ?
    """, (email,))

    row = c.fetchone()
    conn.close()

    if not row:
        return None

    session_id, stored_password = row

    if stored_password == hash_password(password):
        return session_id

    return None


# ==========================================================
# 5. SAVE MESSAGE
# ==========================================================
def save_message(session_id, conversation_id, role, message, timestamp):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    INSERT INTO messages (session_id, conversation_id, role, message, timestamp)
    VALUES (?, ?, ?, ?, ?)
    """, (session_id, conversation_id, role, message, timestamp))

    conn.commit()
    conn.close()


# ==========================================================
# 6. LOAD MESSAGES
# ==========================================================
def load_messages(session_id, conversation_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    SELECT role, message, timestamp
    FROM messages
    WHERE session_id = ? AND conversation_id = ?
    ORDER BY id ASC
    """, (session_id, conversation_id))

    rows = c.fetchall()
    conn.close()

    return rows


# ==========================================================
# 7. LOAD CONVERSATIONS
# ==========================================================
def load_conversations(session_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    SELECT DISTINCT conversation_id
    FROM messages
    WHERE session_id = ?
    ORDER BY id DESC
    """, (session_id,))

    rows = c.fetchall()
    conn.close()

    return [r[0] for r in rows]


# ==========================================================
# 8. GET LAST MESSAGE
# ==========================================================
def get_last_message(session_id, conversation_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    SELECT message
    FROM messages
    WHERE session_id = ? AND conversation_id = ?
    ORDER BY id DESC
    LIMIT 1
    """, (session_id, conversation_id))

    row = c.fetchone()
    conn.close()

    return row[0] if row else "New conversation"