import sqlite3

DB_PATH = "resumes.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            text TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_resume(filename: str, text: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO resumes (filename, text) VALUES (?, ?)",
        (filename, text)
    )
    conn.commit()
    conn.close()


def get_all_resumes():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename, text FROM resumes")
    rows = cursor.fetchall()
    conn.close()
    return rows


def resume_exists(filename: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM resumes WHERE filename = ?", (filename,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def delete_resume_db(filename: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM resumes WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()
