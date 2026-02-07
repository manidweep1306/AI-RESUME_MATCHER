# AI Resume Matcher

An AI-powered resume matching system using NLP, embeddings, and semantic similarity.

## Features
- Resume & JD parsing (PDF, DOCX, TXT)
- Text preprocessing
- Sentence-transformer embeddings
- Cosine similarity matching
- FAISS vector search
- FastAPI backend

## Run API
```bash
uvicorn api.main:app --reload
