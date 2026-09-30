"""
Unit and integration tests for Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser.
"""

import math
import numpy as np
import pytest

from assignment_4.src.climate.models import Region, CropRule
from assignment_4.src.climate.metrics import (
    vector_cosine_similarity,
    validate_against_scipy,
    compute_pairwise_metric_matrices,
)
from assignment_4.src.climate.peaks import (
    detect_rainy_season_peaks,
    generate_multiyear_climate_records,
)


@pytest.fixture
def kampala_region():
    return Region("Kampala", [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130])


@pytest.fixture
def gulu_region():
    return Region("Gulu", [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15])


@pytest.fixture
def mbarara_region():
    return Region("Mbarara", [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90])


def test_region_initialization_and_statistics(kampala_region):
    """Verifies Region attributes, annual total, mean, wettest/driest month, and CV."""
    assert kampala_region.name == "Kampala"
    assert len(kampala_region) == 12
    assert math.isclose(kampala_region.annual_cumulative(), 1600.0)
    assert math.isclose(kampala_region.monthly_mean(), 1600.0 / 12.0)
    wet_m, wet_val = kampala_region.wettest_month()
    dry_m, dry_val = kampala_region.driest_month()
    assert wet_m == "May" and wet_val == 220.0
    assert dry_m == "Sep" and dry_val == 60.0
    assert kampala_region.coefficient_of_variation() > 0


def test_region_validation_boundary():
    """Boundary Case: Invalid length or negative rainfall raises ValueError."""
    with pytest.raises(ValueError):
        Region("InvalidLength", [100, 120])
    with pytest.raises(ValueError):
        Region("NegativeRain", [10, 20, -5, 40, 50, 60, 70, 80, 90, 100, 110, 120])


def test_crop_rule_classification(kampala_region):
    """Verifies agronomic state classification."""
    maize_rule = CropRule("Maize", min_threshold_mm=80.0, max_threshold_mm=180.0)
    assert maize_rule.classify_month(50.0) == "Drought stress"
    assert maize_rule.classify_month(120.0) == "Optimal"
    assert maize_rule.classify_month(220.0) == "Waterlogging risk"

    classes = maize_rule.classify_region(kampala_region)
    assert len(classes) == 12
    assert "Optimal" in classes
    assert "Waterlogging risk" in classes  # May is 220 mm


def test_vector_cosine_vs_scipy(kampala_region, gulu_region):
    """
    Algorithmic Correction: Validates custom vector cosine similarity
    against scipy.spatial.distance.cosine.
    """
    u = kampala_region.monthly_rainfall
    v = gulu_region.monthly_rainfall
    custom_cos, scipy_sim, is_matching = validate_against_scipy(u, v)

    assert is_matching
    assert -1.0 <= custom_cos <= 1.0


def test_cosine_similarity_vs_scalar_math_cos():
    """
    Highlights the previous cohort's algorithmic mistake calling math.cos() on scalar angles
    vs true n-dimensional vector inner product.
    """
    u = np.array([10.0, 20.0, 30.0])
    v = np.array([20.0, 40.0, 60.0])  # Collinear vector
    sim = vector_cosine_similarity(u, v)
    assert math.isclose(sim, 1.0, rel_tol=1e-9)


def test_pairwise_metric_matrices(kampala_region, gulu_region, mbarara_region):
    """Verifies pairwise metric matrices computation."""
    regions = [kampala_region, gulu_region, mbarara_region]
    matrices = compute_pairwise_metric_matrices(regions)

    cos_mat = matrices["cosine_similarity_matrix"]
    pearson_mat = matrices["pearson_correlation_matrix"]
    eucl_mat = matrices["euclidean_distance_matrix"]

    assert cos_mat.shape == (3, 3)
    assert np.allclose(np.diag(cos_mat), 1.0)
    assert np.allclose(np.diag(pearson_mat), 1.0)
    assert np.allclose(np.diag(eucl_mat), 0.0)


def test_modal_peak_detection(kampala_region, gulu_region, mbarara_region):
    """Verifies modal peak detection corresponds to recognized climatological zones."""
    res_gulu = detect_rainy_season_peaks(gulu_region)
    assert res_gulu["modal_classification"] == "Unimodal"
    assert res_gulu["matches_unma_reference"] is True

    res_kla = detect_rainy_season_peaks(kampala_region)
    assert res_kla["modal_classification"] == "Bimodal"
    assert res_kla["matches_unma_reference"] is True

    res_mba = detect_rainy_season_peaks(mbarara_region)
    assert res_mba["modal_classification"] == "Bimodal"
    assert res_mba["matches_unma_reference"] is True


def test_multiyear_climate_records(kampala_region):
    """Extension Case: Verifies 10-year monthly precipitation synthesis."""
    records = generate_multiyear_climate_records(kampala_region, years=10, random_seed=42)
    assert records.shape == (10, 12)
    assert np.all(records > 0.0)
