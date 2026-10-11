"""Local inference and exact additive explanations for calibrated logistic PD."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit, logit

from credit_scoring.config import (
    ARTIFACT_DIR, FEATURE_LABELS, NUMERIC_FEATURES, PAY_STATUS, RAW_FEATURES,
    RISK_BANDS, SCORE_CONFIG,
)
from credit_scoring.data import validate_features


def risk_band(probability: float) -> str:
    if probability < RISK_BANDS["low_upper"]:
        return "LOW"
    if probability < RISK_BANDS["medium_upper"]:
        return "MEDIUM"
    return "HIGH"


def credit_score(probability: float) -> int:
    p = float(np.clip(probability, 1e-9, 1 - 1e-9))
    score = SCORE_CONFIG["base_score"] + SCORE_CONFIG["pdo"] / np.log(2) * np.log(
        (1 - p) / p / SCORE_CONFIG["base_good_bad_odds"]
    )
    return int(np.clip(round(score), SCORE_CONFIG["display_min"], SCORE_CONFIG["display_max"]))


def calibrated_probability(pipeline, calibration: dict, frame: pd.DataFrame) -> np.ndarray:
    linear_score = calibration["slope"] * pipeline.decision_function(frame)
    return expit(linear_score + calibration["intercept"])


class ScoringEngine:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.pipeline = bundle["pipeline"]
        self.calibration = bundle["calibration"]
        self.threshold = bundle["threshold"]
        self.model_version = bundle["model_version"]

    @classmethod
    def load(cls, path: Path = ARTIFACT_DIR / "model.joblib"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                "Model belum tersedia. Jalankan python -m credit_scoring.train"
            )
        # Read the small manifest before unpickling: artifacts use a fixed sklearn version.
        manifest_path = path.with_name("manifest.json")
        if not manifest_path.exists():
            raise FileNotFoundError("manifest.json tidak ditemukan di sebelah model.joblib.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["sklearn_version"] != sklearn.__version__:
            raise RuntimeError(
                f"Model memakai scikit-learn {manifest['sklearn_version']}; "
                f"Python Anda memakai {sklearn.__version__}. "
                "Install requirements.txt atau latih ulang model."
            )
        return cls(joblib.load(path))

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        features = validate_features(frame)
        return self._predict_validated(features)

    def _predict_validated(self, features: pd.DataFrame) -> np.ndarray:
        """Score a frame that has already passed the public input boundary."""
        return calibrated_probability(self.pipeline, self.calibration, features)

    def explain(self, row: pd.DataFrame) -> dict:
        row = validate_features(row)
        if len(row) != 1:
            raise ValueError("Penjelasan membutuhkan tepat satu profil debitur.")
        return self._explain_validated(row)

    def _explain_validated(self, row: pd.DataFrame) -> dict:
        """Explain one validated row without repeating input validation."""
        engineered = self.pipeline.named_steps["features"].transform(row)
        transformed = self.pipeline.named_steps["preprocess"].transform(engineered)[0]
        coefficients = self.pipeline.named_steps["model"].coef_[0]
        reference = self.bundle["reference_mean"]
        slope = self.calibration["slope"]
        contributions = slope * coefficients * (transformed - reference)
        baseline = float(slope * (
            self.pipeline.named_steps["model"].intercept_[0] + coefficients @ reference
        ) + self.calibration["intercept"])
        # ColumnTransformer emits numeric features, then each fitted status category block.
        grouped = {col: float(contributions[i]) for i, col in enumerate(NUMERIC_FEATURES)}
        offset = len(NUMERIC_FEATURES)
        status_encoder = self.pipeline.named_steps["preprocess"].named_transformers_["status"]
        for col, categories in zip(PAY_STATUS, status_encoder.categories_):
            category_count = len(categories)
            grouped[col] = float(contributions[offset:offset + category_count].sum())
            offset += category_count

        def contribution_direction(value: float) -> str:
            if np.isclose(value, 0.0, atol=1e-12):
                return "pengaruh netral pada prediksi"
            if value > 0:
                return "menaikkan risiko model"
            return "menurunkan risiko model"

        factors = sorted([
            {"feature": feature, "label": FEATURE_LABELS[feature],
             "log_odds_contribution": value,
             "direction": contribution_direction(value)}
            for feature, value in grouped.items()
        ], key=lambda factor: abs(factor["log_odds_contribution"]), reverse=True)
        reconstructed = float(expit(baseline + sum(grouped.values())))
        probability = float(self._predict_validated(row)[0])
        if not np.isclose(reconstructed, probability, atol=1e-10):
            raise AssertionError("Kontribusi penjelasan tidak merekonstruksi PD.")
        return {"reference_pd": float(expit(baseline)), "reference_log_odds": baseline,
                "reconstructed_pd": reconstructed, "factors": factors}

    def score(self, payload: dict) -> dict:
        row = validate_features(pd.DataFrame([payload]), exact_columns=True)
        probability = float(self._predict_validated(row)[0])
        outside = [col for col in RAW_FEATURES
                   if not self.bundle["training_ranges"][col]["min"] <= float(row.iloc[0][col])
                   <= self.bundle["training_ranges"][col]["max"]]
        explanation = self._explain_validated(row)
        return {
            "model_version": self.model_version,
            "probability_of_default": probability,
            "horizon": "bulan berikutnya; definisi target historis UCI",
            "credit_score": credit_score(probability),
            "risk_band": risk_band(probability),
            "review_threshold": self.threshold,
            "review_flag": bool(probability >= self.threshold),
            "currency": "NTD", "outside_training_range": outside,
            "explanation": explanation,
        }
