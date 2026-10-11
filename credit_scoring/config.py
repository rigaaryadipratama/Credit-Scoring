from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "credit_card_clients.xls"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
TARGET = "default payment next month"
PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
PAY_STATUS_CODES = tuple(range(-2, 10))
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]
MONEY_COLUMNS = ["LIMIT_BAL", *BILLS, *PAYMENTS]
RAW_FEATURES = ["LIMIT_BAL", *PAY_STATUS, *BILLS, *PAYMENTS]
ENGINEERED = [
    "latest_utilization", "mean_utilization", "payment_balance_proxy",
    "balance_change", "delinquent_months", "longest_delay",
]
NUMERIC_FEATURES = [*MONEY_COLUMNS, *ENGINEERED]
EXCLUDED = ["ID", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
SEED = 42
RISK_BANDS = {"low_upper": 0.10, "medium_upper": 0.30}
SCORE_CONFIG = {"base_score": 600, "base_good_bad_odds": 50, "pdo": 50,
                "display_min": 300, "display_max": 850}
FN_COST = 5.0
FP_COST = 1.0
MAX_INPUT_NTD = 1_000_000_000
FEATURE_LABELS = {
    "LIMIT_BAL": "Limit kredit",
    "latest_utilization": "Rasio tagihan terbaru terhadap limit",
    "mean_utilization": "Rata-rata rasio tagihan terhadap limit",
    "payment_balance_proxy": "Rasio pembayaran terhadap tagihan (proxy)",
    "balance_change": "Perubahan tagihan terhadap limit",
    "delinquent_months": "Jumlah bulan dengan keterlambatan",
    "longest_delay": "Keterlambatan terpanjang",
}
for i, col in enumerate(PAY_STATUS, start=1):
    FEATURE_LABELS[col] = f"Status pembayaran bulan ke-{i}"
for i, col in enumerate(BILLS, start=1):
    FEATURE_LABELS[col] = f"Tagihan bulan ke-{i}"
for i, col in enumerate(PAYMENTS, start=1):
    FEATURE_LABELS[col] = f"Pembayaran bulan ke-{i}"
