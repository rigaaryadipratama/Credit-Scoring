"""Fetch the original UCI dataset if the included raw file has been removed."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
RAW = ROOT / "data" / "raw" / "credit_card_clients.xls"


def main():
    if RAW.exists() and "--force" not in sys.argv:
        print(f"Dataset sudah ada: {RAW}. Gunakan --force untuk mengunduh ulang.")
        return
    with urlopen(URL, timeout=60) as response:
        data = response.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("Unduhan melebihi batas 20 MB.")
    with ZipFile(BytesIO(data)) as archive:
        content = archive.read("default of credit card clients.xls")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_bytes(content)
    provenance = {
        "source": "UCI Default of Credit Card Clients", "creator": "I-Cheng Yeh",
        "doi": "https://doi.org/10.24432/C55S3H", "download_url": URL,
        "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(content).hexdigest(), "modified": False,
    }
    (RAW.parent / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Dataset tersimpan: {RAW} ({len(content):,} bytes)")


if __name__ == "__main__":
    main()
