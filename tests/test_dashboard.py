"""Smoke tests for the Streamlit dashboard and its interaction state."""

import tomllib
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
THEME_PATH = APP_PATH.parent / ".streamlit" / "config.toml"


def relative_luminance(color: str) -> float:
    """Return WCAG relative luminance for a six-digit hex color."""
    channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(first: str, second: str) -> float:
    """Return the WCAG contrast ratio between two hex colors."""
    lighter, darker = sorted(
        [relative_luminance(first), relative_luminance(second)],
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        self.assertEqual(list(self.app.exception), [])

    def test_all_pages_render_without_exceptions(self):
        for page in ["Evaluasi model", "Data & metode", "Simulasi debitur"]:
            with self.subTest(page=page):
                self.app.radio[0].set_value(page).run()
                self.assertEqual(list(self.app.exception), [])

    def test_simulation_result_is_cleared_when_scenario_changes(self):
        self.app.button[0].click().run()
        self.assertEqual(list(self.app.exception), [])
        self.assertEqual(
            [metric.label for metric in self.app.metric],
            [
                "Probabilitas gagal bayar",
                "Skor kredit simulasi",
                "Status peninjauan",
            ],
        )

        self.app.selectbox[0].select(
            "Terlambat dua bulan (skenario stres)"
        ).run()
        self.assertEqual(list(self.app.exception), [])
        self.assertEqual(list(self.app.metric), [])

    def test_payment_status_uses_readable_labels(self):
        editor_frame = self.app.dataframe[0].value
        statuses = set(editor_frame["Kondisi pembayaran"])
        self.assertEqual(statuses, {"Kode khusus dataset (0)"})


class DashboardThemeTests(unittest.TestCase):
    def test_theme_has_readable_text_contrast(self):
        config = tomllib.loads(THEME_PATH.read_text(encoding="utf-8"))
        theme = config["theme"]
        self.assertEqual(theme["base"], "light")

        for background_key in ["backgroundColor", "secondaryBackgroundColor"]:
            with self.subTest(background=background_key):
                ratio = contrast_ratio(theme["textColor"], theme[background_key])
                self.assertGreaterEqual(ratio, 4.5)


if __name__ == "__main__":
    unittest.main()
