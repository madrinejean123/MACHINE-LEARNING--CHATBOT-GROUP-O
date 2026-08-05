# Runs the EXISTING Streamlit app and the new FastAPI /ask API in the same
# Hugging Face Space, on the same public URL — nginx routes between them.
# This replaces the Space's SDK from "streamlit" to "docker" but the Space
# itself, its URL, and its Streamlit UI are unchanged.

FROM python:3.11-slim

# Same apt packages the app already declares in packages.txt (OCR + voice),
# plus nginx to route between the two internal processes.
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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]
