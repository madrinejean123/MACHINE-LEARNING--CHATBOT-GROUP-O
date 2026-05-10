"""
db.py — SQLite storage for Safeguarding Companion chat history
No authentication required; uses session_id isolation.
"""

import sqlite3

DB_NAME = "chat.db"


# ==========================================================
# 1. INITIALISE DATABASE
# ==========================================================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

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
# 2. SAVE MESSAGE
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
# 3. LOAD MESSAGES (FOR ONE USER + ONE CHAT)
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
# 4. LOAD ALL CONVERSATIONS (FOR SIDEBAR)
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
# 5. GET LAST MESSAGE (FOR TITLES)
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