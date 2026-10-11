#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Buat virtual environment dan install requirements.txt terlebih dahulu. Lihat README.md."
  exit 1
fi
exec .venv/bin/python -m streamlit run app.py
