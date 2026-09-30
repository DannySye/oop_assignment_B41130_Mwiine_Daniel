"""
Climate analysis and crop suitability package.
"""

from .models import Region, CropRule, MONTH_NAMES
from .metrics import (
    vector_cosine_similarity,
    validate_against_scipy,
    compute_pairwise_metric_matrices,
)
from .peaks import (
    detect_rainy_season_peaks,
    generate_multiyear_climate_records,
)

__all__ = [
    "Region",
    "CropRule",
    "MONTH_NAMES",
    "vector_cosine_similarity",
    "validate_against_scipy",
    "compute_pairwise_metric_matrices",
    "detect_rainy_season_peaks",
    "generate_multiyear_climate_records",
]
