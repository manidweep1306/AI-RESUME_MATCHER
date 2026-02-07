from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
from pydantic import BaseModel
import numpy as np

from src.database.db import (
    insert_resume,
    get_all_resumes,
    create_table,
    resume_exists
)
from src.embedding.embedding import Embedder
from src.ingestion.resume_parser import extract_resume_text
from src.preprocessing.text_cleaner import clean_text
from src.matching.faiss_ranker import FaissRanker

# ---------------- APP INIT ----------------
app = FastAPI(title="AI Resume Matcher")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

create_table()

embedder = Embedder()
ranker = FaissRanker(dim=384)

STOPWORDS = {"a","an","the","and","or","for","with","to","of","in","on","is","are","was","were"}


class JobDescription(BaseModel):
    text: str


# ---------------- BUILD FAISS FROM DB ----------------
def build_faiss_from_db():
    resumes = get_all_resumes()
    if not resumes:
        return

    embeddings = []
    filenames = []

    for filename, text in resumes:
        emb = embedder.embed(text)
        embeddings.append(emb)
        filenames.append(filename)

    ranker.add(np.array(embeddings), filenames)


build_faiss_from_db()


@app.get("/")
def home():
    return {"message": "AI Resume Matcher API running"}


# ---------------- UPLOAD RESUME ----------------
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):

    if resume_exists(file.filename):
        return {
            "message": "Resume already exists",
            "filename": file.filename
        }

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    raw_text = extract_resume_text(file_path)
    clean = clean_text(raw_text)

    embedding = embedder.embed(clean)

    insert_resume(file.filename, clean)

    ranker.add(
        embeddings=np.array([embedding]),
        filenames=[file.filename]
    )

    return {
        "message": "Resume uploaded and indexed successfully",
        "filename": file.filename
    }


# ---------------- RANK RESUMES ----------------
@app.post("/rank-resumes")
def rank_resumes(job: JobDescription):
    jd_clean = clean_text(job.text)
    jd_embedding = embedder.embed(jd_clean)

    results = ranker.rank(jd_embedding)

    seen = set()
    unique_results = []

    for r in results:
        if r["filename"] not in seen:
            seen.add(r["filename"])
            unique_results.append(r)

    return {"results": unique_results}


# ---------------- EXPLAIN MATCH ----------------
@app.post("/explain-match")
def explain_match(job: JobDescription):
    jd_clean = clean_text(job.text)
    resumes = get_all_resumes()

    if not resumes:
        return {"message": "No resumes found. Upload resumes first."}

    explanations = []
    jd_words = set(jd_clean.lower().split()) - STOPWORDS

    for filename, resume_text in resumes:
        resume_words = set(clean_text(resume_text).lower().split()) - STOPWORDS
        common = list(jd_words.intersection(resume_words))

        explanations.append({
            "filename": filename,
            "matched_keywords": common[:10]
        })

    return {"explanations": explanations}

def rebuild_faiss_index():
    ranker.reset()
    resumes = get_all_resumes()

    if not resumes:
        return

    embeddings = []
    filenames = []

    for filename, text in resumes:
        emb = embedder.embed(text)
        embeddings.append(emb)
        filenames.append(filename)

    ranker.add(np.array(embeddings), filenames)

def reset(self):
    self.index.reset()
    self.filenames = []

from src.database.db import delete_resume_db

@app.delete("/delete-resume/{filename}")
def delete_resume(filename: str):
    filename = filename.strip()

    if not resume_exists(filename):
        return {"message": "Resume not found"}

    delete_resume_db(filename)
    ranker.remove(filename)

    return {"message": f"{filename} deleted successfully"}
