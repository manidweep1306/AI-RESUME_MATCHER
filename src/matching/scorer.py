import numpy as np


def compute_match_score(resume_vec: np.ndarray, jd_vec: np.ndarray) -> float:
    """
    Cosine similarity (dot product because vectors are normalized)
    """
    return float(np.dot(resume_vec, jd_vec))
