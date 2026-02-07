import faiss
import numpy as np
import os
import pickle


class FaissRanker:
    def __init__(self, dim: int, index_path="faiss.index", meta_path="faiss_meta.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        self.index = faiss.IndexFlatL2(dim)
        self.filenames = []

        self.load()

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.filenames = pickle.load(f)

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.filenames, f)

    def reset(self):
        self.index.reset()
        self.filenames = []
        self.save()

    def add(self, embeddings: np.ndarray, filenames: list):
        for emb, fname in zip(embeddings, filenames):
            if fname in self.filenames:
                continue

            self.index.add(np.array([emb]))
            self.filenames.append(fname)

        self.save()

    def rank(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index.ntotal == 0:
            return []

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
            return False

        idx = self.filenames.index(filename)
        self.filenames.pop(idx)

        vectors = self.index.reconstruct_n(0, self.index.ntotal)
        vectors = np.delete(vectors, idx, axis=0)

        self.index.reset()
        if len(vectors) > 0:
            self.index.add(vectors)

        self.save()
        return True
