#!/bin/bash
set -e

python -m src.rag.ingest

uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 &

streamlit run src/frontend/app.py --server.port 8501 --server.address 0.0.0.0
