"""
Metric space comparisons for precipitation vectors:
Cosine similarity, Pearson correlation, and Euclidean distance.
"""

from __future__ import annotations
from typing import Sequence, Dict, Tuple, List
import numpy as np
import scipy.spatial.distance

from .models import Region


def vector_cosine_similarity(u: Sequence[float] | np.ndarray, v: Sequence[float] | np.ndarray) -> float:
    """
    Computes vector cosine similarity between two 1D sequences:
        S_c(u, v) = (u . v) / (||u||_2 * ||v||_2)

    Note: Previous cohorts mistakenly called math.cos(), which calculates scalar trigonometric
    cosine of an angle in radians. This function implements true n-dimensional vector inner product similarity.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    if u_arr.ndim != 1 or v_arr.ndim != 1 or len(u_arr) != len(v_arr):
        raise ValueError("Inputs must be 1D vectors of equal length.")

    norm_u = np.linalg.norm(u_arr)
    norm_v = np.linalg.norm(v_arr)

    if norm_u == 0.0 or norm_v == 0.0:
        raise ValueError("Cannot compute cosine similarity with a zero vector.")

    return float(np.dot(u_arr, v_arr) / (norm_u * norm_v))


def validate_against_scipy(u: Sequence[float], v: Sequence[float]) -> Tuple[float, float, bool]:
    """
    Validates custom vector_cosine_similarity against scipy.spatial.distance.cosine.
    Since scipy computes cosine distance D = 1 - S_c, S_c = 1 - D.
    """
    custom_cos = vector_cosine_similarity(u, v)
    scipy_dist = scipy.spatial.distance.cosine(u, v)
    scipy_sim = float(1.0 - scipy_dist)
    is_matching = bool(np.isclose(custom_cos, scipy_sim, atol=1e-9))
    return custom_cos, scipy_sim, is_matching


def compute_pairwise_metric_matrices(regions: Sequence[Region]) -> Dict[str, Any]:
    """
    Computes pairwise Cosine Similarity, Pearson Correlation, and Euclidean Distance
    matrices across a sequence of Region objects.
    """
    n = len(regions)
    names = [r.name for r in regions]

    cosine_mat = np.zeros((n, n), dtype=np.float64)
    pearson_mat = np.zeros((n, n), dtype=np.float64)
    euclidean_mat = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            u = regions[i].monthly_rainfall
            v = regions[j].monthly_rainfall

            # Cosine similarity
            cosine_mat[i, j] = vector_cosine_similarity(u, v)

            # Pearson correlation coefficient r
            if i == j:
                pearson_mat[i, j] = 1.0
            else:
                corr = np.corrcoef(u, v)[0, 1]
                pearson_mat[i, j] = float(corr)

            # Euclidean distance L2
            euclidean_mat[i, j] = float(np.linalg.norm(u - v))

    explanation = (
        "METRIC SPACE COMPARISON EXPLANATION:\n"
        "1. Cosine Similarity measures the angular alignment between two rainfall vectors in R^12.\n"
        "   Because S_c(u, v) = (u . v) / (||u|| * ||v||), scalar scaling factors cancel out: "
        "   S_c(k * u, v) = S_c(u, v). Thus, two regions with the exact same seasonal rainfall curve "
        "   (e.g., peak in April, dry in July) will yield a cosine similarity near 1.0, even if one "
        "   receives 2,500 mm annually and the other receives only 500 mm!\n"
        "2. Euclidean Distance (L2 norm) measures absolute geometric distance in R^12: "
        "   d(u, v) = sqrt(sum((u_i - v_i)^2)). It directly penalizes differences in rainfall volume.\n"
        "3. Pearson Correlation (r) centers both vectors around their means before computing cosine similarity: "
        "   r(u, v) = S_c(u - u_bar, v - v_bar). It is invariant to both scale and constant vertical offsets."
    )

    return {
        "region_names": names,
        "cosine_similarity_matrix": cosine_mat,
        "pearson_correlation_matrix": pearson_mat,
        "euclidean_distance_matrix": euclidean_mat,
        "metric_explanation": explanation,
    }
