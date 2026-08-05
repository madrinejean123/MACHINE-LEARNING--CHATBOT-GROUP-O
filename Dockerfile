# Runs the EXISTING Streamlit app and the new FastAPI /ask API in the same
# Hugging Face Space, on the same public URL — nginx routes between them.
#
# The two apps get SEPARATE Python virtual environments. This is
# deliberate: Streamlit bundles its own internal Starlette server code
# that is sensitive to the exact starlette version installed, and
# FastAPI/uvicorn need a starlette version of their own. Installing both
# into one shared environment lets pip's resolver pick a starlette that
# satisfies one and silently breaks the other. Two venvs means each app's
# dependencies are resolved completely independently.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    espeak-ng \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- venv 1: the existing Streamlit app, untouched dependency set ---
RUN python -m venv /opt/venv-streamlit
COPY requirements.txt .
RUN /opt/venv-streamlit/bin/pip install --no-cache-dir -r requirements.txt

# --- venv 2: the new FastAPI /ask API, fully isolated ---
RUN python -m venv /opt/venv-api
COPY requirements-api.txt .
RUN /opt/venv-api/bin/pip install --no-cache-dir -r requirements-api.txt

COPY . .

RUN chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]
