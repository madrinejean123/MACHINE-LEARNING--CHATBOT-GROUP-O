#!/bin/sh
set -e

echo "Starting FastAPI (Safeguarding Companion /ask API) on :8000 ..."
uvicorn api:app --host 127.0.0.1 --port 8000 &

echo "Starting Streamlit (chat UI) on :8501 ..."
streamlit run app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --browser.gatherUsageStats false &

echo "Starting nginx on :7860 (public port, routes both) ..."
nginx -c /app/nginx.conf -g "daemon off;"
