from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
from pydantic import BaseModel
import numpy as np

from src.database.db import get_all_resumes
from src.matching.explainer import ResumeExplainer
from src.embedding.embedding import Embedder
from src.ingestion.resume_parser import extract_resume_text
from src.preprocessing.text_cleaner import clean_text
from src.matching.faiss_ranker import FaissRanker

app = FastAPI(title="AI Resume Matcher")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


embedder = Embedder()
ranker = FaissRanker(dim=384)


class JobDescription(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "AI Resume Matcher API running"}


@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_resume_text(file_path)
    clean = clean_text(text)
    embedding = embedder.embed(clean)

    ranker.add(
        embeddings=np.array([embedding]),
        filenames=[file.filename]
    )

    return {"message": "Resume uploaded and indexed successfully", "filename": file.filename}


@app.post("/rank-resumes")
def rank_resumes(job: JobDescription):
    jd_clean = clean_text(job.text)
    jd_embedding = embedder.embed(jd_clean)

    results = ranker.rank(jd_embedding)

    return {"results": results}


explainer = ResumeExplainer()

@app.post("/explain-match")
def explain_match(payload: dict):
    job_text = payload["text"]

    resumes = get_all_resumes()

    if not resumes:
        return {"message": "No resumes uploaded yet"}

    # use top resume (rank 1)
    top_resume = resumes[0]["text"]
    filename = resumes[0]["filename"]

    explanation = explainer.explain(top_resume, job_text)

    return {
        "filename": filename,
        "explanation": explanation
    }
