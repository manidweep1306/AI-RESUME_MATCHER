import faiss
import numpy as np


class FaissRanker:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.filenames = []

    def add(self, embeddings: np.ndarray, filenames: list):
        self.index.add(embeddings)
        self.filenames.extend(filenames)

    def rank(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(np.array([query_embedding]), top_k)

        results = []
        for rank, idx in enumerate(I[0]):
            if idx < len(self.filenames):
                results.append({
                    "rank": rank + 1,
                    "filename": self.filenames[idx],
                    "score": float(D[0][rank])
                })
        return results

    def remove(self, filename: str):
        if filename not in self.filenames:
            return

        idx = self.filenames.index(filename)
        self.filenames.pop(idx)

        # rebuild index
        vectors = self.index.reconstruct_n(0, self.index.ntotal)
        vectors = np.delete(vectors, idx, axis=0)

        self.index.reset()
        if len(vectors) > 0:
            self.index.add(vectors)
