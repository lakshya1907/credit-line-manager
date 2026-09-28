import numpy as np
import pytest

from src.calibrate import calibrate_pd, apply_calibrator


@pytest.fixture
def monotonic_training_data():
    rng = np.random.default_rng(42)
    scores = rng.uniform(0, 1, size=500)
    # y is more likely to be 1 as the raw score increases -> genuinely
    # calibratable relationship.
    y = (rng.uniform(0, 1, size=500) < scores).astype(int)
    return scores, y


def test_isotonic_calibrator_returns_expected_shape(monotonic_training_data):
    scores, y = monotonic_training_data
    calibrator = calibrate_pd(scores, y, method="isotonic")
    kind, obj = calibrator
    assert kind == "isotonic"

    out = apply_calibrator(calibrator, scores)
    assert out.shape == scores.shape


def test_isotonic_calibrator_output_in_unit_interval(monotonic_training_data):
    scores, y = monotonic_training_data
    calibrator = calibrate_pd(scores, y, method="isotonic")
    out = apply_calibrator(calibrator, scores)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_isotonic_calibrator_is_monotonic_nondecreasing(monotonic_training_data):
    scores, y = monotonic_training_data
    calibrator = calibrate_pd(scores, y, method="isotonic")
    test_scores = np.linspace(0, 1, 50)
    out = apply_calibrator(calibrator, test_scores)
    assert np.all(np.diff(out) >= -1e-9)


def test_isotonic_calibrator_clips_out_of_bounds_scores(monotonic_training_data):
    scores, y = monotonic_training_data
    calibrator = calibrate_pd(scores, y, method="isotonic")
    # out_of_bounds="clip" -> values outside the training range should not
    # raise and should clamp to the boundary calibrated values.
    out = apply_calibrator(calibrator, [-5.0, 5.0])
    in_range = apply_calibrator(calibrator, [0.0, 1.0])
    assert out[0] == pytest.approx(in_range[0])
    assert out[1] == pytest.approx(in_range[1])


def test_platt_calibrator_returns_expected_shape_and_range(monotonic_training_data):
    scores, y = monotonic_training_data
    calibrator = calibrate_pd(scores, y, method="platt")
    kind, obj = calibrator
    assert kind == "platt"

    out = apply_calibrator(calibrator, scores)
    assert out.shape == scores.shape
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_apply_calibrator_handles_scalar_and_list_input(monotonic_training_data):
    scores, y = monotonic_training_data
    calibrator = calibrate_pd(scores, y, method="isotonic")
    single = apply_calibrator(calibrator, [0.5])
    assert single.shape == (1,)
