import sqlite3
import numpy as np

DB_PATH = "resumes.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()


def save_resume(text: str, embedding: np.ndarray):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO resumes (text, embedding) VALUES (?, ?)",
        (text, embedding.tobytes())
    )
    conn.commit()
    conn.close()


def load_resumes():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT text, embedding FROM resumes")
    rows = cursor.fetchall()
    conn.close()

    texts = []
    embeddings = []

    for text, blob in rows:
        emb = np.frombuffer(blob, dtype=np.float32)
        texts.append(text)
        embeddings.append(emb)

    return texts, embeddings
