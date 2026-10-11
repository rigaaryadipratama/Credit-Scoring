"""A local CLI, not a web API: python scripts/predict.py"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from credit_scoring.scoring import ScoringEngine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediksi satu profil dari JSON.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "examples" / "applicant.json")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = ScoringEngine.load().score(payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))
