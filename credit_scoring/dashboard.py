"""Streamlit interface for exploring the credit-scoring model.

This module coordinates presentation and user interaction. Model validation,
feature engineering, and inference remain in the core ``credit_scoring`` package.
"""

from __future__ import annotations

import json
from html import escape
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from credit_scoring.config import (
    ARTIFACT_DIR,
    BILLS,
    FEATURE_LABELS,
    MAX_INPUT_NTD,
    PAYMENTS,
    PAY_STATUS,
    REPORT_DIR,
    ROOT,
)
from credit_scoring.scoring import ScoringEngine

Payload = dict[str, int | float]
ScoreResult = dict[str, Any]

STYLE_PATH = ROOT / ".streamlit" / "dashboard.css"
EXAMPLE_PATH = ROOT / "data" / "examples" / "applicant.json"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"
METRICS_PATH = REPORT_DIR / "metrics.json"

MAX_MONEY_VALUE = float(MAX_INPUT_NTD)
MONTH_LABELS = [
    "September 2005 · terbaru",
    "Agustus 2005",
    "Juli 2005",
    "Juni 2005",
    "Mei 2005",
    "April 2005 · terlama",
]

SCENARIOS = {
    "Contoh dasar (kode status 0)": (
        "Profil contoh dengan pola pembayaran yang relatif stabil."
    ),
    "Terlambat dua bulan (skenario stres)": (
        "Seluruh bulan diatur terlambat dua bulan untuk melihat respons model."
    ),
}

STATUS_LABELS = {
    -2: "Kode khusus dataset (-2)",
    -1: "Bayar tepat waktu (-1)",
    0: "Kode khusus dataset (0)",
    **{month: f"Terlambat {month} bulan ({month})" for month in range(1, 9)},
    9: "Terlambat 9+ bulan (9)",
}
STATUS_CODES = {label: code for code, label in STATUS_LABELS.items()}

RISK_LABELS = {
    "LOW": "Rendah",
    "MEDIUM": "Sedang",
    "HIGH": "Tinggi",
}
RISK_CLASSES = {
    "LOW": "risk-low",
    "MEDIUM": "risk-medium",
    "HIGH": "risk-high",
}

PAGES = {
    "Simulasi debitur": {
        "eyebrow": "SIMULATOR RISIKO",
        "title": "Kenali risiko dengan konteks yang jelas",
        "description": (
            "Ubah limit dan riwayat pembayaran, lalu lihat bagaimana model "
            "memperkirakan risiko gagal bayar pada bulan berikutnya."
        ),
    },
    "Evaluasi model": {
        "eyebrow": "KINERJA MODEL",
        "title": "Angka model, tanpa menyembunyikan keterbatasannya",
        "description": (
            "Tinjau kemampuan pemisahan risiko, kalibrasi probabilitas, dan "
            "dampak perubahan batas peninjauan."
        ),
    },
    "Data & metode": {
        "eyebrow": "TRANSPARANSI",
        "title": "Dari data historis hingga sebuah skor",
        "description": (
            "Pelajari sumber data, alur pemodelan, fitur yang digunakan, dan "
            "batas penggunaan dashboard ini."
        ),
    },
}


@st.cache_resource(show_spinner=False)
def load_engine(model_mtime: float, manifest_mtime: float) -> ScoringEngine:
    """Load the model, refreshing the cache when either artifact changes."""
    return ScoringEngine.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_report(report_mtime: float) -> dict[str, Any]:
    """Read evaluation metrics, refreshing when the report is regenerated."""
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_example() -> Payload:
    """Return the bundled applicant example."""
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def apply_dashboard_styles() -> None:
    """Inject the dashboard-specific layout and component styles."""
    if not STYLE_PATH.exists():
        return
    css = STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def load_dashboard_resources() -> tuple[ScoringEngine, dict[str, Any]]:
    """Load model and report together and reject a mismatched pair."""
    required_files = [MODEL_PATH, MANIFEST_PATH, METRICS_PATH]
    missing_files = [path.name for path in required_files if not path.exists()]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Berkas model belum lengkap ({missing}). Jalankan proses training terlebih dahulu."
        )

    engine = load_engine(MODEL_PATH.stat().st_mtime, MANIFEST_PATH.stat().st_mtime)
    report = load_report(METRICS_PATH.stat().st_mtime)
    if report.get("model_version") != engine.model_version:
        raise RuntimeError(
            "Versi model dan laporan evaluasi tidak sama. Jalankan training ulang agar "
            "seluruh artefak tersinkronisasi."
        )
    return engine, report


def scenario_payload(scenario: str) -> Payload:
    """Build the editable starting values for a selected scenario."""
    payload = dict(load_example())
    if scenario == "Terlambat dua bulan (skenario stres)":
        for column in PAY_STATUS:
            payload[column] = 2
    return payload


def history_frame(payload: Payload) -> pd.DataFrame:
    """Present the six-month history using friendly labels."""
    return pd.DataFrame(
        {
            "Bulan": MONTH_LABELS,
            "Kondisi pembayaran": [
                STATUS_LABELS[int(payload[column])] for column in PAY_STATUS
            ],
            "Tagihan (NTD)": [float(payload[column]) for column in BILLS],
            "Pembayaran (NTD)": [float(payload[column]) for column in PAYMENTS],
        }
    )


def payload_from_history(limit: float, history: pd.DataFrame) -> Payload:
    """Translate the editor labels back to the model input schema."""
    payload: Payload = {"LIMIT_BAL": float(limit)}

    for index, column in enumerate(PAY_STATUS):
        label = str(history.iloc[index]["Kondisi pembayaran"])
        if label not in STATUS_CODES:
            raise ValueError(f"Status pembayaran tidak dikenali: {label}")
        payload[column] = STATUS_CODES[label]

    for index, column in enumerate(BILLS):
        payload[column] = float(history.iloc[index]["Tagihan (NTD)"])

    for index, column in enumerate(PAYMENTS):
        payload[column] = float(history.iloc[index]["Pembayaran (NTD)"])

    return payload


def clear_last_scoring() -> None:
    """Hide a result as soon as one of its source inputs changes."""
    st.session_state.pop("last_scoring", None)


def render_page_hero(page: str) -> None:
    """Render a consistent, high-contrast heading for every page."""
    copy = PAGES[page]
    st.markdown(
        f"""
        <section class="dashboard-hero">
            <p class="hero-eyebrow">{escape(copy['eyebrow'])}</p>
            <h1>{escape(copy['title'])}</h1>
            <p class="hero-description">{escape(copy['description'])}</p>
            <div class="hero-tags">
                <span>Logistic Regression</span>
                <span>Data UCI Taiwan</span>
                <span>Horizon 1 bulan</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, description: str | None = None) -> None:
    """Render a compact section title with optional supporting copy."""
    supporting_copy = f"<p>{escape(description)}</p>" if description else ""
    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{escape(title)}</h2>
            {supporting_copy}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(engine: ScoringEngine) -> str:
    """Render navigation and model identity in the sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="brand-mark">CS</div>
                <div>
                    <strong>Credit Scoring Lab</strong>
                    <span>Dashboard pembelajaran</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("PILIH HALAMAN")
        page = st.radio(
            "Navigasi dashboard",
            list(PAGES),
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Model aktif**")
        st.write("Logistic Regression terkalibrasi")
        st.caption(f"Versi {engine.model_version}")
        st.markdown(
            """
            <div class="sidebar-note">
                Model berjalan lokal dan tidak mengirim data input ke API eksternal.
            </div>
            """,
            unsafe_allow_html=True,
        )
    return page


def render_result(result: ScoreResult, payload: Payload) -> None:
    """Render the latest score, review signal, and local explanation."""
    default_probability = float(result["probability_of_default"])
    risk_band = str(result["risk_band"])
    risk_label = RISK_LABELS.get(risk_band, risk_band.title())
    risk_class = RISK_CLASSES.get(risk_band, "risk-medium")

    render_section_heading(
        "Hasil simulasi",
        "Ringkasan ini berasal dari input terakhir yang Anda hitung.",
    )
    st.markdown(
        f"""
        <div class="result-banner">
            <div>
                <span class="result-kicker">ESTIMASI RISIKO BULAN BERIKUTNYA</span>
                <strong>{default_probability:.1%}</strong>
            </div>
            <span class="risk-badge {risk_class}">Risiko {escape(risk_label)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    probability_column, score_column, review_column = st.columns(3)
    probability_column.metric(
        "Probabilitas gagal bayar",
        f"{default_probability:.1%}",
        help=(
            "Estimasi kemungkinan label default pada bulan berikutnya berdasarkan "
            "data historis UCI."
        ),
        border=True,
    )
    score_column.metric(
        "Skor kredit simulasi",
        f"{result['credit_score']}",
        help=(
            "Skala internal proyek 300–850. Skor lebih tinggi berarti "
            "probabilitas default lebih rendah."
        ),
        border=True,
    )
    review_column.metric(
        "Status peninjauan",
        "Perlu review" if result["review_flag"] else "Di bawah batas",
        help="Sinyal untuk pemeriksaan tambahan, bukan keputusan persetujuan kredit.",
        border=True,
    )

    if result["review_flag"]:
        st.warning(
            f"Profil ini melewati batas peninjauan {result['review_threshold']:.0%}. "
            "Gunakan sinyal ini untuk pemeriksaan tambahan, bukan sebagai keputusan otomatis."
        )
    else:
        st.success(
            f"Estimasi berada di bawah batas peninjauan {result['review_threshold']:.0%}. "
            "Hasil ini tetap bukan keputusan persetujuan pinjaman."
        )

    outside_range = result["outside_training_range"]
    if outside_range:
        readable_columns = [
            FEATURE_LABELS.get(column, column) for column in outside_range
        ]
        st.warning(
            "Ada nilai di luar rentang data training: "
            + ", ".join(readable_columns)
            + ". Interpretasikan prediksi dengan lebih hati-hati."
        )

    render_section_heading(
        "Apa yang paling memengaruhi hasil?",
        "Batang ke kanan menaikkan estimasi risiko; batang ke kiri menurunkannya.",
    )
    factors = result["explanation"]["factors"][:8]
    graph_data = pd.DataFrame(factors).sort_values("log_odds_contribution")
    values = graph_data["log_odds_contribution"]

    figure, axis = plt.subplots(figsize=(10, 5.1), layout="constrained")
    figure.patch.set_facecolor("#FFFFFF")
    axis.set_facecolor("#FFFFFF")
    axis.barh(
        graph_data["label"],
        values,
        color=["#C2410C" if value > 0 else "#0F766E" for value in values],
    )
    axis.axvline(0, color="#64748B", linewidth=1)
    axis.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_xlabel("Kontribusi terhadap risiko model", color="#334155")
    axis.tick_params(colors="#334155")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#CBD5E1")
    st.pyplot(
        figure,
        width="stretch",
        alt="Grafik delapan faktor terbesar yang memengaruhi risiko model.",
    )
    plt.close(figure)
    st.caption(
        "Kontribusi menjelaskan perilaku model, bukan hubungan sebab-akibat. "
        "Grafik menampilkan delapan faktor terbesar."
    )

    with st.expander("Lihat seluruh kontribusi dan input mentah"):
        contribution_table = pd.DataFrame(result["explanation"]["factors"])[
            ["label", "log_odds_contribution", "direction"]
        ].rename(
            columns={
                "label": "Faktor",
                "log_odds_contribution": "Kontribusi",
                "direction": "Arah pengaruh",
            }
        )
        st.dataframe(
            contribution_table,
            hide_index=True,
            width="stretch",
            column_config={
                "Kontribusi": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.json(payload)

    download_payload = json.dumps(
        {"input": payload, "result": result},
        indent=2,
        ensure_ascii=False,
    )
    st.download_button(
        "Unduh hasil sebagai JSON",
        download_payload,
        file_name="credit_scoring_result.json",
        mime="application/json",
        width="stretch",
    )


def render_simulation_page(engine: ScoringEngine) -> None:
    """Render editable applicant history and score it on demand."""
    render_page_hero("Simulasi debitur")
    st.info(
        "Dashboard memakai limit kredit dan enam bulan riwayat pembayaran. "
        "Dataset ini tidak memiliki informasi pendapatan atau nominal pinjaman baru."
    )

    render_section_heading(
        "Siapkan profil simulasi",
        "Mulai dari contoh, kemudian sesuaikan nilainya dengan skenario yang ingin diuji.",
    )
    scenario = st.selectbox(
        "Titik awal simulasi",
        list(SCENARIOS),
        help="Pilihan ini hanya mengisi nilai awal dan tidak mengubah model.",
        on_change=clear_last_scoring,
    )
    st.caption(SCENARIOS[scenario])
    payload = scenario_payload(scenario)

    limit = st.number_input(
        "Limit kredit (NTD)",
        min_value=1.0,
        max_value=MAX_MONEY_VALUE,
        value=float(payload["LIMIT_BAL"]),
        step=1_000.0,
        format="%.0f",
        key=f"limit_{scenario}",
        on_change=clear_last_scoring,
        help="Batas kredit aktif nasabah dalam New Taiwan Dollar.",
    )

    with st.expander("Panduan membaca status pembayaran"):
        st.markdown(
            """
            - **Bayar tepat waktu (-1):** pembayaran dilakukan sesuai jadwal.
            - **Terlambat 1–8 bulan:** jumlah bulan keterlambatan yang tercatat.
            - **Terlambat 9+ bulan (9):** keterlambatan sembilan bulan atau lebih.
            - **Kode -2 dan 0:** kategori asli dataset yang tidak dijelaskan secara
              rinci dalam dokumentasi publik UCI. Model mempertahankannya tanpa
              menebak maknanya.
            """
        )

    history = st.data_editor(
        history_frame(payload),
        hide_index=True,
        width="stretch",
        height=286,
        num_rows="fixed",
        disabled=["Bulan"],
        key=f"history_{scenario}",
        on_change=clear_last_scoring,
        column_config={
            "Bulan": st.column_config.TextColumn("Bulan", width="medium"),
            "Kondisi pembayaran": st.column_config.SelectboxColumn(
                "Kondisi pembayaran",
                options=list(STATUS_LABELS.values()),
                required=True,
                width="large",
            ),
            "Tagihan (NTD)": st.column_config.NumberColumn(
                "Tagihan (NTD)",
                min_value=-MAX_MONEY_VALUE,
                max_value=MAX_MONEY_VALUE,
                step=100.0,
                required=True,
                format="%.0f",
            ),
            "Pembayaran (NTD)": st.column_config.NumberColumn(
                "Pembayaran (NTD)",
                min_value=0.0,
                max_value=MAX_MONEY_VALUE,
                step=100.0,
                required=True,
                format="%.0f",
            ),
        },
    )

    if st.button("Hitung risiko dan skor", type="primary", width="stretch"):
        try:
            current_payload = payload_from_history(limit, history)
            result = engine.score(current_payload)
            st.session_state["last_scoring"] = (result, current_payload)
        except (ValueError, TypeError, OverflowError) as error:
            clear_last_scoring()
            st.error(f"Ada input yang perlu diperiksa: {error}")

    if "last_scoring" in st.session_state:
        render_result(*st.session_state["last_scoring"])


def render_evaluation_charts(image_path: str) -> None:
    """Split the six-panel report image into readable, responsive tabs."""
    image = plt.imread(image_path)
    height, width = image.shape[:2]
    column_edges = [0, width // 3, 2 * width // 3, width]
    row_edges = [int(height * 0.03), height // 2, height]
    chart_specs = [
        ("ROC", 0, 0, "Kurva ROC pada data test."),
        ("Precision–recall", 0, 1, "Kurva precision–recall pada data test."),
        ("Kalibrasi", 0, 2, "Grafik kalibrasi probabilitas pada data test."),
        ("Confusion matrix", 1, 0, "Confusion matrix pada batas review terpilih."),
        ("Distribusi risiko", 1, 1, "Distribusi probabilitas menurut outcome aktual."),
        ("Perbandingan", 1, 2, "Perbandingan model dengan baseline konstan."),
    ]

    tabs = st.tabs([spec[0] for spec in chart_specs])
    for tab, (_, row, column, alt_text) in zip(tabs, chart_specs):
        with tab:
            crop = image[
                row_edges[row] : row_edges[row + 1],
                column_edges[column] : column_edges[column + 1],
            ]
            st.image(crop, width="stretch", alt=alt_text)


def render_evaluation_page(
    engine: ScoringEngine,
    report: dict[str, Any],
) -> None:
    """Render held-out metrics and an interactive validation-threshold view."""
    render_page_hero("Evaluasi model")
    test_metrics = report["test"]

    render_section_heading(
        "Kinerja pada data test",
        "Data test dipisahkan sejak awal dan tidak dipakai untuk memilih model atau batas review.",
    )
    metric_columns = st.columns(4)
    metric_definitions = [
        (
            "ROC-AUC",
            "roc_auc",
            "Kemampuan model mengurutkan profil berisiko dan tidak berisiko.",
        ),
        (
            "Average precision",
            "pr_auc",
            "Ringkasan precision–recall untuk kelas default.",
        ),
        (
            "Brier score",
            "brier",
            "Kesalahan probabilitas; semakin kecil semakin baik.",
        ),
        (
            "Recall default",
            "recall",
            "Proporsi kasus default yang berhasil ditandai.",
        ),
    ]
    for column, (label, key, help_text) in zip(metric_columns, metric_definitions):
        column.metric(
            label,
            f"{test_metrics[key]:.3f}",
            help=help_text,
            border=True,
        )

    st.caption(
        f"{test_metrics['rows']:,} baris test · prevalensi default "
        f"{test_metrics['default_rate']:.1%} · batas review "
        f"{test_metrics['threshold']:.0%}."
    )

    evaluation_image = REPORT_DIR / "evaluation.png"
    if evaluation_image.exists():
        render_evaluation_charts(str(evaluation_image))
        st.caption(
            "Pilih tab untuk membaca setiap grafik pada ukuran yang lebih nyaman."
        )
    else:
        st.warning(
            "Grafik evaluasi belum tersedia. Jalankan training untuk membuatnya kembali."
        )

    comparison = pd.DataFrame(
        {
            "Metrik": [
                "ROC-AUC",
                "Average precision",
                "Brier score",
                "Log loss",
                "Accuracy",
                "Recall",
                "Precision",
                "KS",
            ],
            "Model terkalibrasi": [
                test_metrics[key]
                for key in [
                    "roc_auc",
                    "pr_auc",
                    "brier",
                    "log_loss",
                    "accuracy",
                    "recall",
                    "precision",
                    "ks",
                ]
            ],
            "Baseline konstan": [
                report["baseline_test"][key]
                for key in [
                    "roc_auc",
                    "pr_auc",
                    "brier",
                    "log_loss",
                    "accuracy",
                    "recall",
                    "precision",
                    "ks",
                ]
            ],
        }
    )
    st.dataframe(
        comparison,
        hide_index=True,
        width="stretch",
        column_config={
            "Model terkalibrasi": st.column_config.NumberColumn(format="%.3f"),
            "Baseline konstan": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    render_section_heading(
        "Eksplorasi batas peninjauan",
        "Bagian ini memakai data validation dan tidak mengubah model maupun hasil test di atas.",
    )
    thresholds_path = REPORT_DIR / "validation_thresholds.csv"
    if not thresholds_path.exists():
        st.warning(
            "Data eksplorasi batas belum tersedia. Jalankan training untuk membuatnya kembali."
        )
        return

    review_threshold = st.slider(
        "Batas probabilitas untuk review",
        min_value=0.01,
        max_value=0.99,
        value=float(engine.threshold),
        step=0.01,
        format="percent",
    )
    thresholds = pd.read_csv(thresholds_path)
    selected = thresholds.iloc[
        (thresholds["threshold"] - review_threshold).abs().argmin()
    ]

    recall_column, review_column, cost_column = st.columns(3)
    recall_column.metric(
        "Recall validation",
        f"{selected['recall']:.1%}",
        border=True,
    )
    review_column.metric(
        "Profil masuk review",
        f"{selected['review_rate']:.1%}",
        border=True,
    )
    cost_column.metric(
        "Biaya ilustratif / 1.000",
        f"{selected['illustrative_cost_per_1000']:.1f}",
        border=True,
    )
    st.caption(
        "Biaya ilustratif memberi bobot 5 untuk false negative dan 1 untuk false positive. "
        "Angka ini bukan estimasi kerugian finansial bank."
    )

    chart_data = thresholds.rename(
        columns={
            "threshold": "Batas review",
            "recall": "Recall",
            "precision": "Precision",
            "review_rate": "Proporsi review",
        }
    )
    st.line_chart(
        chart_data,
        x="Batas review",
        y=["Recall", "Precision", "Proporsi review"],
        x_label="Batas probabilitas",
        y_label="Nilai metrik",
        color=["#1D4ED8", "#0F766E", "#C2410C"],
        width="stretch",
        height=380,
    )

    with st.expander("Lihat kandidat model pada tahap validation"):
        st.json(report["validation_candidates"])


def render_method_page(report: dict[str, Any]) -> None:
    """Render dataset provenance, modeling flow, and responsible-use notes."""
    render_page_hero("Data & metode")

    render_section_heading(
        "Tentang dataset",
        "Konteks data menentukan apa yang boleh—dan tidak boleh—disimpulkan dari model.",
    )
    source_column, scope_column = st.columns(2)
    with source_column:
        st.markdown(
            """
            #### Sumber

            **Default of Credit Card Clients** berisi 30.000 pemegang kartu kredit
            Taiwan. Riwayat pembayaran mencakup April–September 2005, dengan target
            default pada bulan berikutnya.

            [Buka dataset UCI][uci-dataset]
            · [Lisensi CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

            [uci-dataset]: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
            """
        )
    with scope_column:
        st.markdown(
            """
            #### Cakupan fitur

            Model memakai limit kredit, status pembayaran, nilai tagihan, dan nilai
            pembayaran. **ID, jenis kelamin, pendidikan, status pernikahan, dan umur**
            tidak dipakai sebagai fitur prediksi.

            Menghapus demografi tidak otomatis membuktikan bahwa model bebas bias.
            """
        )

    render_section_heading("Alur pemodelan")
    st.markdown(
        """
        <div class="process-grid">
            <div>
                <span>01</span><strong>Validasi data</strong>
                <p>Periksa skema, tipe, dan nilai input.</p>
            </div>
            <div>
                <span>02</span><strong>Pisahkan data</strong>
                <p>Train, calibration, validation, dan test.</p>
            </div>
            <div>
                <span>03</span><strong>Latih model</strong>
                <p>Rekayasa fitur, standardisasi, dan regresi logistik.</p>
            </div>
            <div>
                <span>04</span><strong>Kalibrasi</strong>
                <p>Selaraskan skor model dengan probabilitas historis.</p>
            </div>
            <div>
                <span>05</span><strong>Evaluasi</strong>
                <p>Laporkan hasil pada test yang tetap terpisah.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    split_table = pd.DataFrame(report["splits"]).T.rename(
        columns={
            "rows": "Baris",
            "default_rate": "Proporsi default",
            "unique_profiles": "Profil unik",
        }
    )
    split_table.index.name = "Bagian data"
    st.dataframe(
        split_table,
        width="stretch",
        column_config={
            "Proporsi default": st.column_config.NumberColumn(format="percent"),
        },
    )

    eda_image = REPORT_DIR / "eda_train.png"
    if eda_image.exists():
        st.image(
            str(eda_image),
            caption="Eksplorasi data training.",
            width="stretch",
        )
    else:
        st.warning(
            "Grafik eksplorasi belum tersedia. Jalankan training untuk membuatnya kembali."
        )

    render_section_heading("Batas penggunaan")
    st.warning(
        "Dashboard ini dibuat untuk pembelajaran dan portofolio. Penggunaan di dunia nyata "
        "memerlukan data yang relevan, validasi lintas waktu, audit fairness, kebijakan "
        "keputusan yang terdokumentasi, serta monitoring berkelanjutan."
    )
    st.markdown(
        """
        Dataset tidak memuat pendapatan maupun nominal pinjaman baru. Karena itu, model ini
        tidak boleh langsung dipindahkan untuk menilai aplikasi pinjaman baru. Gunakan dataset
        aplikasi yang memiliki fitur dan outcome sesuai konteks tersebut.

        **Paper acuan:** Dastile, Celik, dan Potsane (2020),
        *Statistical and machine learning models in credit scoring: A systematic
        literature survey*.
        [Buka DOI paper](https://doi.org/10.1016/j.asoc.2020.106263).
        """
    )

    model_card_path = REPORT_DIR / "model_card.md"
    with st.expander("Baca model card lengkap"):
        if model_card_path.exists():
            st.markdown(model_card_path.read_text(encoding="utf-8"))
        else:
            st.warning(
                "Model card belum tersedia. Jalankan training untuk membuatnya kembali."
            )


def render_footer() -> None:
    """Render a quiet purpose reminder."""
    st.markdown(
        """
        <footer class="dashboard-footer">
            Credit Scoring Lab · Dibuat untuk eksplorasi model secara transparan,
            bukan untuk keputusan kredit otomatis.
        </footer>
        """,
        unsafe_allow_html=True,
    )


def render_startup_error(error: Exception) -> None:
    """Show a recovery path instead of a raw stack trace."""
    st.error(f"Dashboard belum dapat dimuat: {error}")
    st.markdown("Jalankan perintah berikut dari folder proyek, lalu muat ulang halaman:")
    st.code(
        "python -m pip install -r requirements.txt\n"
        "python -m credit_scoring.train\n"
        "python -m streamlit run app.py",
        language="bash",
    )


def main() -> None:
    """Configure and render the dashboard."""
    st.set_page_config(
        page_title="Credit Scoring Lab",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_dashboard_styles()

    try:
        engine, report = load_dashboard_resources()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        render_startup_error(error)
        st.stop()

    page = render_sidebar(engine)
    if page == "Simulasi debitur":
        render_simulation_page(engine)
    elif page == "Evaluasi model":
        render_evaluation_page(engine, report)
    else:
        render_method_page(report)
    render_footer()
