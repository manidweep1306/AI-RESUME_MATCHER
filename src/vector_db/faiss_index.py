import faiss
import numpy as np


class ResumeFaissIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.resume_texts = []

    def add_resume(self, embedding: np.ndarray, resume_text: str):
        self.index.add(embedding.reshape(1, -1))
        self.resume_texts.append(resume_text)

    def search(self, jd_embedding: np.ndarray, top_k: int = 3):
        scores, indices = self.index.search(jd_embedding.reshape(1, -1), top_k)
        results = []

        for idx, score in zip(indices[0], scores[0]):
            results.append({
                "score": float(score),
                "resume_text": self.resume_texts[idx]
            })

        return results
