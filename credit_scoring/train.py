"""Run from the project root: python -m credit_scoring.train"""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
import warnings

from credit_scoring.config import (
    ARTIFACT_DIR, DATA_PATH, EXCLUDED, FN_COST, FP_COST, RAW_FEATURES,
    REPORT_DIR, RISK_BANDS, ROOT, SCORE_CONFIG, SEED, TARGET,
)
from credit_scoring.data import dataset_sha256, load_dataset, profile_groups, split_dataset
from credit_scoring.evaluation import calculate_metrics, choose_threshold
from credit_scoring.features import make_pipeline
from credit_scoring.scoring import calibrated_probability


def fit_calibration(pipeline, X, y) -> dict:
    logits = pipeline.decision_function(X).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logits, y)
    slope = float(calibrator.coef_[0, 0])
    if slope <= 0:
        raise ValueError("Kalibrasi membalik ranking risiko; periksa data dan model.")
    return {"method": "sigmoid_holdout", "slope": slope,
            "intercept": float(calibrator.intercept_[0])}


def evaluation_plot(y, probability, baseline, metrics, output):
    color = "#167d91"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), layout="constrained")
    fpr, tpr, _ = roc_curve(y, probability)
    axes[0, 0].plot(fpr, tpr, color=color, lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], "--", color="#9aa6b1", label="Baseline konstan")
    axes[0, 0].set(title="ROC: data test", xlabel="False positive rate", ylabel="True positive rate")
    axes[0, 0].legend(loc="lower right")
    precision, recall, _ = precision_recall_curve(y, probability)
    axes[0, 1].plot(recall, precision, color=color, lw=2, label=f"AP = {metrics['pr_auc']:.3f}")
    axes[0, 1].axhline(np.mean(y), ls="--", color="#9aa6b1", label="Prevalensi test")
    axes[0, 1].set(title="Precision-recall: data test", xlabel="Recall default", ylabel="Precision default")
    axes[0, 1].legend()
    observed, predicted = calibration_curve(y, probability, n_bins=10, strategy="quantile")
    axes[0, 2].plot(predicted, observed, "o-", color=color, label="Model terkalibrasi")
    axes[0, 2].plot([0, 1], [0, 1], "--", color="#9aa6b1", label="Ideal")
    axes[0, 2].set(title="Kalibrasi: data test", xlabel="Rata-rata PD prediksi", ylabel="Proporsi default aktual")
    axes[0, 2].legend()
    matrix = np.array(metrics["confusion_matrix"])
    axes[1, 0].imshow(matrix, cmap="Blues")
    for (row, col), number in np.ndenumerate(matrix):
        axes[1, 0].text(col, row, str(number), ha="center", va="center", fontsize=17,
                        color="white" if number > matrix.max() / 2 else "#18344d")
    axes[1, 0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-default", "Default"],
                   yticklabels=["Non-default", "Default"], xlabel="Prediksi", ylabel="Aktual",
                   title=f"Confusion matrix (cutoff {metrics['threshold']:.2f})")
    axes[1, 1].hist(np.asarray(probability)[np.asarray(y) == 0], bins=np.linspace(0, 1, 26),
                    alpha=.65, density=True, color=color, label="Non-default")
    axes[1, 1].hist(np.asarray(probability)[np.asarray(y) == 1], bins=np.linspace(0, 1, 26),
                    alpha=.55, density=True, color="#e99340", label="Default")
    axes[1, 1].axvline(metrics["threshold"], color="#29394d", ls="--")
    axes[1, 1].set(title="Distribusi PD: data test", xlabel="Probability of default", ylabel="Density")
    axes[1, 1].legend()
    names = ["ROC-AUC", "PR-AUC (AP)", "Brier (lebih kecil lebih baik)"]
    keys = ["roc_auc", "pr_auc", "brier"]
    x = np.arange(3)
    axes[1, 2].barh(x + .18, [baseline[k] for k in keys], height=.32, color="#c3cdd7", label="Baseline konstan")
    axes[1, 2].barh(x - .18, [metrics[k] for k in keys], height=.32, color=color, label="Logistic Regression")
    axes[1, 2].set(yticks=x, yticklabels=names, xlim=(0, 1), title="Perbandingan pada test")
    axes[1, 2].legend(loc="upper right")
    fig.suptitle("Credit scoring | evaluasi holdout | UCI Taiwan 2005", fontsize=18)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def eda_plot(frame, output):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    counts = frame[TARGET].value_counts().sort_index()
    axes[0].bar(["Non-default", "Default"], counts.values, color=["#167d91", "#e99340"])
    axes[0].set(title="Kelas target pada train", ylabel="Jumlah debitur")
    for i, value in enumerate(counts.values):
        axes[0].text(i, value + counts.max() * .02, f"{value:,}", ha="center")
    axes[0].set_ylim(0, counts.max() * 1.15)
    axes[1].hist(frame["LIMIT_BAL"] / 1000, bins=35, color="#167d91", edgecolor="white")
    axes[1].set(title="Distribusi limit pada train", xlabel="Limit (ribu NTD)", ylabel="Jumlah debitur")
    grouped = frame.groupby("PAY_0")[TARGET].agg(["mean", "count"])
    risks = grouped["mean"]
    axes[2].bar(risks.index.astype(str), risks.values, color="#167d91")
    for i, (_, group) in enumerate(grouped.iterrows()):
        axes[2].text(i, group["mean"] + .025, f"n={int(group['count']):,}",
                     ha="center", va="bottom", fontsize=7.5, rotation=35)
    axes[2].set(title="Status terbaru dan default pada train", xlabel="Kode asli PAY_0", ylabel="Proporsi default", ylim=(0, 1.16))
    fig.suptitle("EDA menggunakan data train; hubungan ini bersifat asosiasi", fontsize=14)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def model_card(report: dict) -> str:
    test = report["test"]
    baseline = report["baseline_test"]
    split_lines = "\n".join(
        f"| {name} | {row['rows']:,} | {row['default_rate']:.2%} |"
        for name, row in report["splits"].items()
    )
    return f"""# Model card - Credit Scoring v1

## Tujuan dan populasi
Estimasi probabilitas label default pembayaran bulan berikutnya untuk pemegang
kartu kredit pada dataset historis Taiwan 2005. Ini proyek portofolio/pembelajaran,
bukan estimasi risiko kredit konsumen Indonesia atau model underwriting pinjaman baru.
Target mengikuti label UCI, bukan definisi buatan '90 hari terlambat' atau PD 12 bulan.
Semua nilai uang memakai NTD.

## Data dan hak penggunaan
I-Cheng Yeh (2009), Default of Credit Card Clients, UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C55S3H. Lisensi data: CC BY 4.0.
30.000 baris asal, 23 fitur asal, April-September 2005; target bulan berikutnya.
File mentah disertakan tanpa perubahan. SHA-256: `{report['dataset_sha256']}`.
19 fitur pembayaran/keuangan dipakai. ID, SEX, EDUCATION, MARRIAGE, dan AGE dikeluarkan.
Mengeluarkan fitur demografis tidak membuktikan model adil; proxy bias masih mungkin.
Data hanya pelanggan yang sudah memperoleh kredit sehingga selection bias tetap ada.

## Pembagian dan pencegahan kebocoran
Seed {SEED}. Sekitar 60% train, 15% calibration, 10% validation, 15% test.
Profil dengan 19 fitur identik, termasuk yang targetnya berbeda, berada di satu split.
Jumlah grup yang menyeberangi split: 0. Rasio aktual mengikuti ukuran grup.
Imputer/scaler/koefisien hanya fit pada train. Kalibrasi sigmoid hanya fit pada
calibration. Pemilihan kandidat dan cutoff hanya memakai validation.
Test digunakan setelah pilihan model dan cutoff dibekukan.
Split acak per profil digunakan karena satu kohort ini tidak memungkinkan
pengujian lintas waktu. Ini tidak membuktikan kinerja pada data masa depan.

| Split | Baris | Proporsi default |
| --- | ---: | ---: |
{split_lines}

## Model dan fitur
Logistic Regression, C=1, regularisasi L2, lbfgs. Kandidat: class_weight=None
dan balanced, keduanya dikalibrasi. Kandidat dipilih berdasarkan log loss validation
terendah (cost validation menjadi tie-breaker). Model terpilih: `{report['selected_model']}`.
Nominal uang memakai signed log1p; fitur numerik distandardisasi. Status pembayaran
dikodekan one-hot -2 sampai 9. Kode -2 dan 0 dipertahankan sebagai kategori asli;
deskripsi publik UCI tidak mendefinisikan keduanya secara rinci.
Fitur tambahan: rasio tagihan/limit, proxy pembayaran/tagihan, perubahan tagihan,
jumlah bulan terlambat, dan keterlambatan terpanjang. Proxy bukan coverage cicilan.
Rasio ekstrem dibatasi secara deterministik di features.py; tidak belajar dari test.

## Hasil test
| Metrik | Logistic terkalibrasi | Baseline probabilitas konstan |
| --- | ---: | ---: |
| ROC-AUC | {test['roc_auc']:.4f} | {baseline['roc_auc']:.4f} |
| PR-AUC / Average Precision | {test['pr_auc']:.4f} | {baseline['pr_auc']:.4f} |
| Brier score (lebih kecil lebih baik) | {test['brier']:.4f} | {baseline['brier']:.4f} |
| Log loss (lebih kecil lebih baik) | {test['log_loss']:.4f} | {baseline['log_loss']:.4f} |
| Recall default | {test['recall']:.4f} | {baseline['recall']:.4f} |
| Precision default | {test['precision']:.4f} | {baseline['precision']:.4f} |

Cutoff flag review: {test['threshold']:.2f}, dipilih dengan grid 0.01-0.99 untuk
meminimalkan cost ilustratif FN:{FN_COST:g}, FP:{FP_COST:g} pada validation.
Bobot ini bukan estimasi kerugian bank. Tidak memasukkan LGD, EAD, pendapatan,
kapasitas review, atau biaya operasional. Flag bukan keputusan approve/reject.
Confusion matrix test, baris aktual 0/1 dan kolom prediksi 0/1:
`{test['confusion_matrix']}`. Accuracy: {test['accuracy']:.4f}, KS: {test['ks']:.4f}.

## PD, skor, dan penjelasan
PD berasal dari sigmoid(slope * logit_model + intercept) pada kalibrasi terpisah.
Tidak ada jaminan PD terkalibrasi untuk populasi lain atau tahun lain.
Skor demo = 600 + 50/ln(2) * ln(((1-PD)/PD)/50), ditampilkan pada rentang 300-850.
Skor 600 berarti good:bad odds 50:1. Ini skala proyek sendiri, bukan FICO/SLIK.
Band demo: LOW <10%, MEDIUM 10%-<30%, HIGH >=30%. Band tidak sama dengan cutoff review.
Penjelasan adalah kontribusi linear pada log-odds terkalibrasi relatif rata-rata
fitur transformasi train. Seluruh kontribusi + intercept merekonstruksi PD tepat.
Penjelasan menunjukkan asosiasi model, bukan sebab, rekomendasi perilaku, atau
adverse-action reason codes yang tervalidasi untuk produksi.

## Batas dan pengembangan
Belum divalidasi dengan data bank Indonesia, split lintas waktu, audit fairness,
reject inference, monitoring drift, atau validasi independen. Tidak ada fitur
income, loan_amount, employment_length: dataset ini memang tidak memuatnya.
Untuk underwriting pinjaman baru, pilih data application dengan outcome/horizon jelas.
Jangan gunakan pendapatan fiktif untuk mengisi fitur yang tidak tersedia.
Joblib hanya dimuat dari artefak proyek yang dipercaya; jangan unggah model asing.
Versi model: `{report['model_version']}`. scikit-learn: {report['sklearn_version']}.

## Referensi paper
Dastile, X., Celik, T., & Potsane, M. (2020). Statistical and machine learning models
in credit scoring: A systematic literature survey. Applied Soft Computing, 91, 106263.
https://doi.org/10.1016/j.asoc.2020.106263.
Proyek menerapkan baseline transparan, evaluasi imbalance, dan transparansi yang
dibahas survey; proyek ini bukan reproduksi seluruh eksperimen dalam paper.
"""


def train(data_path=DATA_PATH, artifact_dir=ARTIFACT_DIR, report_dir=REPORT_DIR) -> dict:
    data_path, artifact_dir, report_dir = map(Path, (data_path, artifact_dir, report_dir))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(data_path)
    splits = split_dataset(frame)
    parts = {name: (frame.iloc[idx][RAW_FEATURES], frame.iloc[idx][TARGET].to_numpy())
             for name, idx in splits.items()}
    candidates = []
    for name, weight in [("logistic_standard", None), ("logistic_balanced", "balanced")]:
        print(f"Fit {name} pada {len(parts['train'][1]):,} baris train...", flush=True)
        pipeline = make_pipeline(weight)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit(*parts["train"])
        calibration = fit_calibration(pipeline, *parts["calibration"])
        probabilities = calibrated_probability(pipeline, calibration, parts["validation"][0])
        threshold, curve = choose_threshold(parts["validation"][1], probabilities)
        metrics = calculate_metrics(parts["validation"][1], probabilities, threshold)
        raw_metrics = calculate_metrics(parts["validation"][1], pipeline.predict_proba(parts["validation"][0])[:, 1], threshold)
        candidates.append({"name": name, "pipeline": pipeline, "calibration": calibration,
                           "threshold": threshold, "validation": metrics,
                           "raw_validation": raw_metrics, "curve": curve})
    best = min(candidates, key=lambda row: (
        row["validation"]["log_loss"], row["validation"]["illustrative_cost_per_1000"],
    ))
    print(f"Pilihan validation: {best['name']}; cutoff review {best['threshold']:.2f}", flush=True)
    # All choices are frozen above. Only the chosen model is now scored on test.
    pipeline = best["pipeline"]
    test_prob = calibrated_probability(pipeline, best["calibration"], parts["test"][0])
    test_metrics = calculate_metrics(parts["test"][1], test_prob, best["threshold"])
    prevalence = float(parts["train"][1].mean())
    base_threshold, _ = choose_threshold(parts["validation"][1], np.full(len(parts["validation"][1]), prevalence))
    baseline = calculate_metrics(parts["test"][1], np.full(len(parts["test"][1]), prevalence), base_threshold)
    training_transform = pipeline.named_steps["preprocess"].transform(
        pipeline.named_steps["features"].transform(parts["train"][0])
    )
    checksum = dataset_sha256(data_path)
    version = f"uci-lr-v1-{checksum[:10]}"
    bundle = {
        "pipeline": pipeline, "calibration": best["calibration"], "threshold": best["threshold"],
        "reference_mean": training_transform.mean(axis=0), "model_version": version,
        "training_ranges": {col: {"min": float(parts["train"][0][col].min()),
                                  "max": float(parts["train"][0][col].max())} for col in RAW_FEATURES},
    }
    split_details = {name: {"rows": int(len(idx)), "default_rate": float(frame.iloc[idx][TARGET].mean()),
                            "unique_profiles": int(profile_groups(frame.iloc[idx]).nunique())}
                     for name, idx in splits.items()}
    report = {
        "model_version": version, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
        "dataset_sha256": checksum, "source_rows": int(len(frame)), "seed": SEED,
        "raw_features": RAW_FEATURES, "excluded_columns": EXCLUDED,
        "cross_split_profile_overlap": 0, "splits": split_details,
        "selected_model": best["name"], "calibration": best["calibration"],
        "risk_bands": RISK_BANDS, "score_scale": SCORE_CONFIG,
        "illustrative_cost": {"false_negative": FN_COST, "false_positive": FP_COST},
        "selection_criterion": "lowest validation log_loss; validation cost as tie-breaker",
        "validation_candidates": [{"name": c["name"], "calibrated": c["validation"],
                                   "uncalibrated": c["raw_validation"], "calibration": c["calibration"]}
                                  for c in candidates],
        "validation": best["validation"], "test": test_metrics, "baseline_test": baseline,
    }
    joblib.dump(bundle, artifact_dir / "model.joblib", compress=3)
    (artifact_dir / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(best["curve"]).drop(columns="confusion_matrix").to_csv(report_dir / "validation_thresholds.csv", index=False)
    membership = frame[["ID"]].copy()
    membership["split"] = ""
    for name, idx in splits.items():
        membership.loc[idx, "split"] = name
    membership.to_csv(report_dir / "split_membership.csv", index=False)
    eda = {"train_rows": len(parts["train"][1]),
           "train_default_rate": prevalence,
           "train_missing_values": frame.iloc[splits["train"]][RAW_FEATURES].isna().sum().to_dict(),
           "train_numeric_summary": frame.iloc[splits["train"]][RAW_FEATURES].describe().round(3).to_dict()}
    (report_dir / "eda_summary.json").write_text(json.dumps(eda, indent=2), encoding="utf-8")
    (report_dir / "model_card.md").write_text(model_card(report), encoding="utf-8")
    evaluation_plot(parts["test"][1], test_prob, baseline, test_metrics, report_dir / "evaluation.png")
    eda_plot(frame.iloc[splits["train"]], report_dir / "eda_train.png")
    print(f"Test ROC-AUC={test_metrics['roc_auc']:.4f}, AP={test_metrics['pr_auc']:.4f}, Brier={test_metrics['brier']:.4f}")
    print(f"Model: {artifact_dir / 'model.joblib'}\nLaporan: {report_dir / 'model_card.md'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latih credit scoring tanpa API.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--reports", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    train(args.data, args.artifacts, args.reports)
