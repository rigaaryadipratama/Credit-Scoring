"""Meaningful end-to-end checks using the included real dataset and saved model."""

import json
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

from credit_scoring.config import RAW_FEATURES, REPORT_DIR, ROOT, SCORE_CONFIG, TARGET
from credit_scoring.data import load_dataset, profile_groups, split_dataset, validate_features
from credit_scoring.evaluation import calculate_metrics, choose_threshold
from credit_scoring.scoring import ScoringEngine, credit_score, risk_band


class CreditScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_dataset()
        cls.splits = split_dataset(cls.frame)
        cls.engine = ScoringEngine.load()
        cls.report = json.loads((REPORT_DIR / "metrics.json").read_text())
        cls.sample = json.loads((ROOT / "data/examples/applicant.json").read_text())

    def test_split_has_no_shared_profiles_or_rows(self):
        groups = profile_groups(self.frame)
        seen_groups, seen_rows = set(), set()
        for positions in self.splits.values():
            current_groups = set(groups.iloc[positions])
            self.assertFalse(seen_groups & current_groups)
            self.assertFalse(seen_rows & set(positions))
            seen_groups |= current_groups
            seen_rows |= set(positions)
            self.assertEqual(self.frame.iloc[positions][TARGET].nunique(), 2)
        self.assertEqual(len(seen_rows), len(self.frame))

    def test_saved_test_metrics_are_reproducible(self):
        test = self.frame.iloc[self.splits["test"]]
        probabilities = self.engine.predict_probability(test[RAW_FEATURES])
        observed = calculate_metrics(test[TARGET], probabilities, self.engine.threshold)
        for metric in ["roc_auc", "pr_auc", "brier", "log_loss", "recall", "precision"]:
            self.assertAlmostEqual(observed[metric], self.report["test"][metric], places=10)
        self.assertEqual(observed["confusion_matrix"], self.report["test"]["confusion_matrix"])

    def test_local_explanation_reconstructs_calibrated_probability(self):
        rows = self.frame.iloc[self.splits["test"][:16]][RAW_FEATURES]
        for i in range(len(rows)):
            row = rows.iloc[[i]]
            explanation = self.engine.explain(row)
            reconstructed = expit(explanation["reference_log_odds"] + sum(
                factor["log_odds_contribution"] for factor in explanation["factors"]
            ))
            self.assertAlmostEqual(reconstructed, self.engine.predict_probability(row)[0], places=10)

    def test_demo_score_increases_when_pd_decreases(self):
        scores = [credit_score(p) for p in np.linspace(0.001, 0.999, 100)]
        self.assertTrue(all(a >= b for a, b in zip(scores, scores[1:])))
        self.assertEqual(credit_score(1 / 51), SCORE_CONFIG["base_score"])
        self.assertEqual(credit_score(0), SCORE_CONFIG["display_max"])
        self.assertEqual(credit_score(1), SCORE_CONFIG["display_min"])

    def test_risk_band_boundaries(self):
        self.assertEqual(risk_band(.09999), "LOW")
        self.assertEqual(risk_band(.10), "MEDIUM")
        self.assertEqual(risk_band(.29999), "MEDIUM")
        self.assertEqual(risk_band(.30), "HIGH")

    def test_invalid_input_and_target_leakage_rejected(self):
        cases = [dict(self.sample, LIMIT_BAL=0), dict(self.sample, PAY_AMT1=-1),
                 dict(self.sample, PAY_0=2.5), dict(self.sample, PAY_0=10),
                 dict(self.sample, BILL_AMT1=float("nan")),
                 dict(self.sample, LIMIT_BAL=float("inf")),
                 dict(self.sample, LIMIT_BAL=True), dict(self.sample, target=1)]
        missing = self.sample.copy()
        missing.pop("PAY_AMT6")
        cases.append(missing)
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises((ValueError, TypeError)):
                    self.engine.score(case)

    def test_negative_credit_balance_is_accepted(self):
        result = self.engine.score(dict(self.sample, BILL_AMT1=-1000))
        self.assertTrue(0 < result["probability_of_default"] < 1)

    def test_demographic_changes_do_not_affect_predictions(self):
        row = self.frame.iloc[[self.splits["test"][0]]].copy()
        before = self.engine.predict_probability(row)[0]
        row["SEX"], row["EDUCATION"], row["MARRIAGE"], row["AGE"], row["ID"] = 9, 9, 9, 99, -1
        self.assertEqual(before, self.engine.predict_probability(row)[0])

    def test_output_schema_and_review_threshold(self):
        result = self.engine.score(self.sample)
        self.assertEqual(result["review_flag"], result["probability_of_default"] >= result["review_threshold"])
        self.assertEqual(result["currency"], "NTD")
        self.assertEqual(len(result["explanation"]["factors"]), 25)
        json.dumps(result, allow_nan=False)

    def test_out_of_training_range_is_flagged(self):
        result = self.engine.score(dict(self.sample, LIMIT_BAL=100000000))
        self.assertIn("LIMIT_BAL", result["outside_training_range"])

    def test_validation_cutoff_matches_saved_policy(self):
        validation = self.frame.iloc[self.splits["validation"]]
        probabilities = self.engine.predict_probability(validation[RAW_FEATURES])
        threshold, _ = choose_threshold(validation[TARGET], probabilities)
        self.assertEqual(threshold, self.engine.threshold)


if __name__ == "__main__":
    unittest.main()
