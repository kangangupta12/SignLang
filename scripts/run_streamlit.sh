#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "--docker" ]; then
  docker run --rm -it -p 8501:8501 signlang:latest
else
  streamlit run app/streamlit_app.py
fi
