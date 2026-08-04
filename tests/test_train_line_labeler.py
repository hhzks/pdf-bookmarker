"""Tests for training/train_line_labeler.py's sklearn-free parts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import train_line_labeler as trainer


def test_early_stopping_is_off():
    """sklearn turns it on above 10k samples, and it starves this detector.

    It stops on the loss over a random 10% of rows; headings are 1.4% of them,
    so it quit at 99 of 300 iterations. Fitting to the cap is worth +1.7 title
    F1 on the test documents (0.7631 -> 0.7797) and +4.6 on validation.
    """
    assert trainer.ESTIMATOR_PARAMS["early_stopping"] is False


def test_the_trees_are_wider_than_the_default():
    """63 leaves against sklearn's 31: chosen on the validation documents."""
    assert trainer.ESTIMATOR_PARAMS["max_leaf_nodes"] == 63


def test_the_class_weighting_survives():
    """1.4% positives: without this the detector collapses to all-negative."""
    assert trainer.ESTIMATOR_PARAMS["class_weight"] == "balanced"


def test_threshold_tuning_finds_the_obvious_cut():
    probabilities = [0.9, 0.8, 0.2, 0.1]
    truth = [1, 1, 0, 0]
    threshold, f1 = trainer.tune_threshold(
        __import__("numpy").asarray(probabilities), __import__("numpy").asarray(truth)
    )
    assert 0.2 < threshold <= 0.8
    assert f1 == 1.0
