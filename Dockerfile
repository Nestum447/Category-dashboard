FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models so the Space starts faster
RUN python -c "from transformers import pipeline; \
    pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment'); \
    pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')"

COPY . .

EXPOSE 7860

ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HF_HOME=/app/.cache/huggingface

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
