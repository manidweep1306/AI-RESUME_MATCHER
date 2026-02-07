from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
from pydantic import BaseModel
import numpy as np

from src.database.db import (
    insert_resume,
    get_all_resumes,
    create_table,
    resume_exists,
    delete_resume_db
)

from src.embedding.embedding import Embedder
from src.ingestion.resume_parser import extract_resume_text
from src.preprocessing.text_cleaner import clean_text
from src.matching.faiss_ranker import FaissRanker

app = FastAPI(title="AI Resume Matcher")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STOPWORDS = {"a","an","the","and","or","for","with","to","of","in","on","is","are","was","were"}

create_table()

embedder = Embedder()
ranker = FaissRanker(dim=384)


class JobDescription(BaseModel):
    text: str


# -------- Build FAISS from DB --------
def build_faiss_from_db():
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


build_faiss_from_db()


@app.get("/")
def home():
    return {"message": "AI Resume Matcher API running"}


# -------- Upload Resume --------
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):

    if resume_exists(file.filename):
        return {"message": "Resume already exists", "filename": file.filename}

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    raw_text = extract_resume_text(file_path)
    clean = clean_text(raw_text)

    embedding = embedder.embed(clean)

    insert_resume(file.filename, clean)
    ranker.add(np.array([embedding]), [file.filename])

    return {"message": "Resume uploaded successfully", "filename": file.filename}


# -------- Rank Resumes --------
@app.post("/rank-resumes")
def rank_resumes(job: JobDescription):
    jd_clean = clean_text(job.text)
    jd_embedding = embedder.embed(jd_clean)

    results = ranker.rank(jd_embedding)

    seen = set()
    unique = []
    for r in results:
        if r["filename"] not in seen:
            seen.add(r["filename"])
            unique.append(r)

    return {"results": unique}


# -------- Explain Match --------
@app.post("/explain-match")
def explain_match(job: JobDescription):
    jd_clean = clean_text(job.text)
    resumes = get_all_resumes()

    if not resumes:
        return {"message": "No resumes found"}

    jd_words = set(jd_clean.lower().split()) - STOPWORDS
    explanations = []

    for filename, resume_text in resumes:
        resume_words = set(clean_text(resume_text).lower().split()) - STOPWORDS
        common = list(jd_words.intersection(resume_words))

        explanations.append({
            "filename": filename,
            "matched_keywords": common[:10]
        })

    return {"explanations": explanations}


# -------- Delete Resume --------
@app.delete("/delete-resume/{filename}")
def delete_resume(filename: str):
    filename = filename.strip()

    if not resume_exists(filename):
        return {"message": "Resume not found"}

    delete_resume_db(filename)
    ranker.remove(filename)

    return {"message": f"{filename} deleted successfully"}
