# Multi-lingual Sentiment & Emotion Dashboard
# Ver en Streamlit Multi-lingual Sentiment & Emotion Dashboard
https://category-dashboard-eihwcops9yz9thmfbnzgqj.streamlit.app/

[![Open in HF Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/Reethika30/sentiment-emotion-dashboard) ![Python](https://img.shields.io/badge/Python-3.11+-blue) ![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red) ![License](https://img.shields.io/badge/License-MIT-green)

🚀 **Live Demo:** https://huggingface.co/spaces/Reethika30/sentiment-emotion-dashboard

Interactive Streamlit dashboard that performs **multi-lingual sentiment** and **emotion classification**
on free-form text or batch CSVs, powered by HuggingFace Transformers.

---

## Features

| Tab | Capability |
|------|-----------|
| **Single Text** | Paste any text → top sentiment + top emotion + confidence + full distribution charts |
| **Batch CSV** | Upload CSV with `text` column → KPI tiles, pie/bar charts, confidence histogram, sentiment×emotion heatmap, downloadable result CSV |
| **Compare** | Enter up to 5 short texts → side-by-side sentiment & emotion bar chart |

## Models

| Task | Model | Notes |
|------|-------|-------|
| Sentiment | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | 3-class (positive/neutral/negative), 8 languages |
| Emotion | `j-hartmann/emotion-english-distilroberta-base` | 7 emotions: joy, anger, fear, sadness, surprise, disgust, neutral |

## Supported Languages (Sentiment)

🇺🇸 English · 🇪🇸 Spanish · 🇫🇷 French · 🇩🇪 German · 🇮🇹 Italian · 🇵🇹 Portuguese · 🇸🇦 Arabic · 🇮🇳 Hindi

## Quick Start

```bash
git clone https://github.com/Reethika30/sentiment-emotion-dashboard.git
cd sentiment-emotion-dashboard
pip install -r requirements.txt
streamlit run app.py
```

First run downloads ~600MB of model weights.

## Architecture

```
┌────────────┐   ┌──────────────┐   ┌────────────────────┐   ┌───────────────┐
│  User text │──▶│  Streamlit   │──▶│  XLM-RoBERTa       │──▶│  Sentiment    │
│  / CSV     │   │  UI (3 tabs) │   │  DistilRoBERTa     │──▶│  + Emotion    │
└────────────┘   └──────────────┘   └────────────────────┘   └───────┬───────┘
                                                                      │
                       ┌──────────────────────────────────────────────┘
                       ▼
                ┌─────────────────────┐
                │  Plotly dashboards  │
                │  + downloadable CSV │
                └─────────────────────┘
```

## Output Schema (Batch CSV)

| Column | Type | Description |
|--------|------|-------------|
| `text` | str | Original input text |
| `sentiment` | str | `positive` / `neutral` / `negative` |
| `sentiment_confidence` | float | Probability of predicted sentiment |
| `emotion` | str | One of 7 emotion labels |
| `emotion_confidence` | float | Probability of predicted emotion |

## Use Cases

- **Customer feedback triage** — auto-classify reviews and surveys
- **Social listening** — monitor brand mentions across languages
- **Support ticket prioritisation** — surface angry / fearful messages first
- **Content moderation** — flag highly negative or angry content
- **Marketing analytics** — track emotional engagement across campaigns

## Project Structure

```
sentiment-emotion-dashboard/
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
├── Dockerfile          # HuggingFace Spaces deployment
├── README.md           # This file
└── HF_SPACE_README.md  # HuggingFace Space metadata
```

## Tech Stack

- **NLP:** HuggingFace Transformers (XLM-RoBERTa, DistilRoBERTa)
- **UI:** Streamlit
- **Visualisation:** Plotly Express
- **Data:** Pandas
- **Deployment:** Docker on HuggingFace Spaces

## License

MIT License — free to use, modify, and distribute.

---

**Author:** [Sree Reethika Kasanagottu](https://github.com/Reethika30) | [LinkedIn](https://www.linkedin.com/in/sree-reethika/)
