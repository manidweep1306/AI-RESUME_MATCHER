import faiss
import numpy as np


class FaissRanker:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity (inner product)
        self.filenames = []

    def add(self, embeddings: np.ndarray, filenames: list[str]):
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.filenames.extend(filenames)

    def rank(self, query_embedding: np.ndarray, top_k: int = 5):
        query = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            results.append({
                "rank": i + 1,
                "filename": self.filenames[idx],
                "score": round(float(scores[0][i]), 3)
            })

        return results
