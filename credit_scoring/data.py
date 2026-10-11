"""Read the original UCI file and keep identical feature profiles in one split."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_scoring.config import (
    BILLS,
    DATA_PATH,
    MAX_INPUT_NTD,
    PAYMENTS,
    PAY_STATUS,
    PAY_STATUS_CODES,
    RAW_FEATURES,
    SEED,
    TARGET,
)


def validate_features(frame: pd.DataFrame, *, exact_columns: bool = False) -> pd.DataFrame:
    """Validate units/codes and preserve negative bills (credit balances are valid)."""
    missing = set(RAW_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Kolom wajib belum ada: {sorted(missing)}")
    if exact_columns and set(frame.columns) != set(RAW_FEATURES):
        raise ValueError("Input hanya boleh berisi 19 kolom fitur pada contoh input.")
    if frame.empty:
        raise ValueError("Input harus memiliki setidaknya satu baris.")
    result = frame[RAW_FEATURES].copy()
    for col in RAW_FEATURES:
        if pd.api.types.is_bool_dtype(result[col]):
            raise ValueError(f"{col} harus berupa angka, bukan boolean.")
        result[col] = pd.to_numeric(result[col], errors="raise")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Input mengandung angka kosong, NaN, atau infinite.")
    if (result["LIMIT_BAL"] <= 0).any():
        raise ValueError("LIMIT_BAL harus lebih besar dari 0, dalam NTD.")
    if (result[PAYMENTS] < 0).any().any():
        raise ValueError("PAY_AMT1 sampai PAY_AMT6 tidak boleh negatif.")
    if not result[PAY_STATUS].isin(PAY_STATUS_CODES).all().all():
        raise ValueError("Status pembayaran harus berupa integer -2 sampai 9.")
    if (
        result[["LIMIT_BAL", *BILLS, *PAYMENTS]].abs() > MAX_INPUT_NTD
    ).any().any():
        raise ValueError("Nilai uang melebihi batas input demo (1 miliar NTD).")
    return result


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}. "
            "Jalankan python scripts/download_data.py"
        )
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_excel(path, header=1, engine="xlrd")
    frame = frame.copy()
    frame[RAW_FEATURES] = validate_features(frame)
    if "ID" not in frame or TARGET not in frame:
        raise ValueError("Dataset wajib memiliki ID dan kolom target UCI.")
    if frame["ID"].isna().any() or frame["ID"].duplicated().any():
        raise ValueError("ID debitur harus unik dan terisi.")
    if not frame[TARGET].isin([0, 1]).all() or frame[TARGET].nunique() != 2:
        raise ValueError("Target harus berisi kedua kelas 0 dan 1, tanpa missing value.")
    return frame.reset_index(drop=True)


def dataset_sha256(path: Path = DATA_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile_groups(frame: pd.DataFrame) -> pd.Series:
    # A feature-identical profile, even with conflicting targets, never crosses splits.
    return pd.util.hash_pandas_object(frame[RAW_FEATURES], index=False).rename("group")


def split_dataset(frame: pd.DataFrame, seed: int = SEED) -> dict[str, np.ndarray]:
    groups = profile_groups(frame)
    grouped = pd.DataFrame({"group": groups, "target": frame[TARGET]})
    group_targets = grouped.groupby("group")["target"].mean().ge(0.5).astype(int)

    def divide(ids, test_size):
        return train_test_split(
            ids, test_size=test_size, random_state=seed,
            stratify=group_targets.loc[ids].to_numpy(),
        )

    train, remainder = divide(group_targets.index.to_numpy(), 0.40)
    development, test = divide(remainder, 0.375)
    calibration, validation = divide(development, 0.40)
    split_groups = {"train": train, "calibration": calibration,
                    "validation": validation, "test": test}
    splits = {name: np.flatnonzero(groups.isin(ids).to_numpy())
              for name, ids in split_groups.items()}
    seen = set()
    for name, positions in splits.items():
        current = set(groups.iloc[positions])
        if current & seen:
            raise AssertionError(f"Kebocoran profil pada split {name}.")
        seen |= current
        if frame.iloc[positions][TARGET].nunique() != 2:
            raise ValueError(f"Split {name} tidak memiliki kedua kelas.")
    if sum(map(len, splits.values())) != len(frame):
        raise AssertionError("Pembagian data tidak lengkap.")
    return splits
