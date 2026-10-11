"""The same feature transformations are used by training and the dashboard."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_scoring.config import (
    BILLS,
    ENGINEERED,
    MONEY_COLUMNS,
    NUMERIC_FEATURES,
    PAYMENTS,
    PAY_STATUS,
    PAY_STATUS_CODES,
)


class CreditFeatures(BaseEstimator, TransformerMixin):
    """Deterministic log transforms and ratios; no statistics learned from test data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X.copy()
        bills = frame[BILLS].clip(lower=0)
        limit = frame["LIMIT_BAL"]
        frame["latest_utilization"] = (bills[BILLS[0]] / limit).clip(0, 5)
        frame["mean_utilization"] = (bills.mean(axis=1) / limit).clip(0, 5)
        # A statistical proxy, not a minimum-payment or installment coverage ratio.
        frame["payment_balance_proxy"] = (
            frame[PAYMENTS].mean(axis=1) / (bills.mean(axis=1) + 1)
        ).clip(0, 5)
        frame["balance_change"] = (
            (frame[BILLS[0]] - frame[BILLS[-1]]) / limit
        ).clip(-3, 3)
        frame["delinquent_months"] = (frame[PAY_STATUS] >= 1).sum(axis=1)
        frame["longest_delay"] = frame[PAY_STATUS].max(axis=1).clip(lower=0)
        # Negative bill balances remain negative after a signed log transform.
        for col in MONEY_COLUMNS:
            frame[col] = np.sign(frame[col]) * np.log1p(frame[col].abs())
        return frame[[*NUMERIC_FEATURES, *PAY_STATUS]]


def make_pipeline(class_weight=None) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("status", OneHotEncoder(
            categories=[list(PAY_STATUS_CODES) for _ in PAY_STATUS],
            handle_unknown="error", sparse_output=False,
        ), PAY_STATUS),
    ], remainder="drop", verbose_feature_names_out=True)
    return Pipeline([
        ("features", CreditFeatures()),
        ("preprocess", preprocess),
        ("model", LogisticRegression(
            C=1.0, class_weight=class_weight, max_iter=3000,
            solver="lbfgs", random_state=42,
        )),
    ])
