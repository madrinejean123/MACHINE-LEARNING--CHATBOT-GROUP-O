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

    # USERS TABLE
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        session_id TEXT UNIQUE NOT NULL
    )
    """)

    # MESSAGES TABLE
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
def _hash(password: str):
    return hashlib.sha256(password.encode()).hexdigest()


# ==========================================================
# 3. CREATE USER (REGISTER)
# ==========================================================
def create_user(email: str, password: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    session_id = str(uuid.uuid4())

    try:
        c.execute("""
        INSERT INTO users (email, password, session_id)
        VALUES (?, ?, ?)
        """, (email.lower().strip(), _hash(password), session_id))

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()


# ==========================================================
# 4. VERIFY USER (LOGIN)
# ==========================================================
def verify_user(email: str, password: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    SELECT password, session_id
    FROM users
    WHERE email = ?
    """, (email.lower().strip(),))

    row = c.fetchone()
    conn.close()

    if not row:
        return None

    stored_password, session_id = row

    if stored_password == _hash(password):
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
# 8. LAST MESSAGE
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