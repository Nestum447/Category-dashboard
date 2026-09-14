```python
"""
Multi-lingual Sentiment & Emotion Analysis Dashboard
=====================================================
Streamlit app powered by HuggingFace Transformers.

Modes:
  - Single text   : paste any text → sentiment + emotion + category + confidence
  - Batch CSV     : upload CSV with a 'text' column → bulk analysis + category dashboards
  - Live compare  : run several texts side-by-side

Models:
  - Sentiment : cardiffnlp/twitter-xlm-roberta-base-sentiment
  - Emotion   : j-hartmann/emotion-english-distilroberta-base
  - Category  : MoritzLaurer/mDeBERTa-v3-base-mnli-xnli

Author: Sree Reethika Kasanagottu
Repo  : https://github.com/Reethika30/sentiment-emotion-dashboard
"""

from __future__ import annotations

import io
from typing import List, Dict

import pandas as pd
import plotly.express as px
import streamlit as st
from transformers import pipeline


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Sentiment & Emotion Dashboard",
    page_icon="💬",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

SENTIMENT_MODEL = (
    "cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

EMOTION_MODEL = (
    "j-hartmann/emotion-english-distilroberta-base"
)

CATEGORY_MODEL = (
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)


# -----------------------------------------------------------------------------
# Business categories
# -----------------------------------------------------------------------------

CATEGORIES = [
    "Costo o precio",
    "Calidad",
    "Entrega",
    "Servicio y atención",
    "Producto",
    "Comunicación",
    "Otros",
]


# -----------------------------------------------------------------------------
# Colors
# -----------------------------------------------------------------------------

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "neutral": "#95a5a6",
    "negative": "#e74c3c",
}


EMOTION_COLORS = {
    "joy": "#f1c40f",
    "anger": "#e74c3c",
    "fear": "#9b59b6",
    "sadness": "#3498db",
    "surprise": "#1abc9c",
    "disgust": "#27ae60",
    "neutral": "#95a5a6",
}


# -----------------------------------------------------------------------------
# Cached model loaders
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading sentiment model…")
def load_sentiment():
    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        top_k=None,
    )


@st.cache_resource(show_spinner="Loading emotion model…")
def load_emotion():
    return pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        top_k=None,
    )


@st.cache_resource(show_spinner="Loading category model…")
def load_category():
    """
    Zero-shot classification used to classify customer comments
    into business categories such as Cost, Quality, Delivery, etc.
    """
    return pipeline(
        "zero-shot-classification",
        model=CATEGORY_MODEL,
    )


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _normalise_label(label: str) -> str:
    """Normalize model labels."""
    return label.lower().strip()


# -----------------------------------------------------------------------------
# Single text analysis
# -----------------------------------------------------------------------------

def analyse_one(
    text: str,
    sent_pipe,
    emo_pipe,
    cat_pipe,
) -> Dict:
    """
    Run sentiment, emotion and category analysis
    on a single text.
    """

    clean_text = str(text).strip()

    if not clean_text:
        return {
            "text": text,
            "sentiment": "neutral",
            "sentiment_confidence": 0.0,
            "emotion": "neutral",
            "emotion_confidence": 0.0,
            "category": "Otros",
            "category_confidence": 0.0,
            "_sent_dist": {},
            "_emo_dist": {},
            "_category_dist": {},
        }

    # Limit text size to avoid excessive model memory usage
    model_text = clean_text[:512]

    # -------------------------------------------------------------------------
    # Sentiment
    # -------------------------------------------------------------------------

    sent_raw = sent_pipe(model_text)[0]

    # Some Transformers versions return a list of dictionaries,
    # while others can return nested lists.
    if sent_raw and isinstance(sent_raw[0], list):
        sent_raw = sent_raw[0]

    sent_top = max(
        sent_raw,
        key=lambda x: x["score"],
    )

    # -------------------------------------------------------------------------
    # Emotion
    # -------------------------------------------------------------------------

    emo_raw = emo_pipe(model_text)[0]

    if emo_raw and isinstance(emo_raw[0], list):
        emo_raw = emo_raw[0]

    emo_top = max(
        emo_raw,
        key=lambda x: x["score"],
    )

    # -------------------------------------------------------------------------
    # Category
    # -------------------------------------------------------------------------

    cat_raw = cat_pipe(
        model_text,
        candidate_labels=CATEGORIES,
        multi_label=False,
    )

    category = cat_raw["labels"][0]
    category_confidence = float(cat_raw["scores"][0])

    category_distribution = {
        label: float(score)
        for label, score in zip(
            cat_raw["labels"],
            cat_raw["scores"],
        )
    }

    # -------------------------------------------------------------------------
    # Return result
    # -------------------------------------------------------------------------

    return {
        "text": text,

        "sentiment": _normalise_label(
            sent_top["label"]
        ),

        "sentiment_confidence": round(
            float(sent_top["score"]),
            4,
        ),

        "emotion": _normalise_label(
            emo_top["label"]
        ),

        "emotion_confidence": round(
            float(emo_top["score"]),
            4,
        ),

        "category": category,

        "category_confidence": round(
            category_confidence,
            4,
        ),

        "_sent_dist": {
            _normalise_label(d["label"]): float(d["score"])
            for d in sent_raw
        },

        "_emo_dist": {
            _normalise_label(d["label"]): float(d["score"])
            for d in emo_raw
        },

        "_category_dist": category_distribution,
    }


# -----------------------------------------------------------------------------
# Batch analysis
# -----------------------------------------------------------------------------

def analyse_batch(
    texts: List[str],
    sent_pipe,
    emo_pipe,
    cat_pipe,
) -> pd.DataFrame:

    rows = []

    progress = st.progress(
        0.0,
        text="Analysing…",
    )

    n = len(texts)

    if n == 0:
        progress.empty()
        return pd.DataFrame()

    for i, text in enumerate(texts):

        if not isinstance(text, str) or not text.strip():
            continue

        row = analyse_one(
            text,
            sent_pipe,
            emo_pipe,
            cat_pipe,
        )

        # Remove detailed distributions from the
        # main DataFrame.
        row.pop("_sent_dist", None)
        row.pop("_emo_dist", None)
        row.pop("_category_dist", None)

        rows.append(row)

        progress.progress(
            (i + 1) / n,
            text=f"Analysing {i + 1}/{n}",
        )

    progress.empty()

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

def render_sidebar() -> None:

    st.sidebar.header("ℹ️ About")

    st.sidebar.markdown(
        """
        **Multi-lingual Sentiment & Emotion Dashboard**

        ### Sentiment
        3-class:
        - Positive
        - Neutral
        - Negative

        XLM-RoBERTa supports multiple languages.

        ### Emotion
        7 classes:
        - Joy
        - Anger
        - Fear
        - Sadness
        - Surprise
        - Disgust
        - Neutral

        ### Business Categories
        - Costo o precio
        - Calidad
        - Entrega
        - Servicio y atención
        - Producto
        - Comunicación
        - Otros

        **Tech:** HuggingFace Transformers · Streamlit · Plotly
        """
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        "[GitHub Repo](https://github.com/Reethika30/sentiment-emotion-dashboard) · "
        "[Author](https://github.com/Reethika30)"
    )


# -----------------------------------------------------------------------------
# TAB 1 - Single text
# -----------------------------------------------------------------------------

def tab_single(
    sent_pipe,
    emo_pipe,
    cat_pipe,
) -> None:

    st.subheader("Single Text Analysis")

    sample = (
        "I absolutely loved the new product launch — "
        "it exceeded all my expectations!"
    )

    text = st.text_area(
        "Enter text "
        "(en, es, fr, de, it, pt, ar, hi)",
        value=sample,
        height=140,
    )

    if st.button(
        "Analyse",
        type="primary",
        key="single_btn",
    ):

        if not text.strip():

            st.warning(
                "Please enter some text."
            )

            return

        with st.spinner(
            "Running models…"
        ):

            result = analyse_one(
                text,
                sent_pipe,
                emo_pipe,
                cat_pipe,
            )

        col1, col2, col3 = st.columns(3)

        # ---------------------------------------------------------------------
        # Sentiment
        # ---------------------------------------------------------------------

        with col1:

            st.metric(
                "Sentiment",
                result["sentiment"].title(),
                f'{result["sentiment_confidence"] * 100:.1f}% confidence',
            )

            sent_df = pd.DataFrame(
                [
                    {
                        "label": k,
                        "score": v,
                    }
                    for k, v in result["_sent_dist"].items()
                ]
            ).sort_values(
                "score",
                ascending=True,
            )

            fig = px.bar(
                sent_df,
                x="score",
                y="label",
                orientation="h",
                color="label",
                color_discrete_map=SENTIMENT_COLORS,
                title="Sentiment distribution",
            )

            fig.update_layout(
                showlegend=False,
                height=260,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        # ---------------------------------------------------------------------
        # Emotion
        # ---------------------------------------------------------------------

        with col2:

            st.metric(
                "Emotion",
                result["emotion"].title(),
                f'{result["emotion_confidence"] * 100:.1f}% confidence',
            )

            emo_df = pd.DataFrame(
                [
                    {
                        "label": k,
                        "score": v,
                    }
                    for k, v in result["_emo_dist"].items()
                ]
            ).sort_values(
                "score",
                ascending=True,
            )

            fig = px.bar(
                emo_df,
                x="score",
                y="label",
                orientation="h",
                color="label",
                color_discrete_map=EMOTION_COLORS,
                title="Emotion distribution",
            )

            fig.update_layout(
                showlegend=False,
                height=260,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        # ---------------------------------------------------------------------
        # Category
        # ---------------------------------------------------------------------

        with col3:

            st.metric(
                "Categoría",
                result["category"],
                f'{result["category_confidence"] * 100:.1f}% confidence',
            )

            cat_df = pd.DataFrame(
                [
                    {
                        "category": k,
                        "score": v,
                    }
                    for k, v in result["_category_dist"].items()
                ]
            ).sort_values(
                "score",
                ascending=True,
            )

            fig = px.bar(
                cat_df,
                x="score",
                y="category",
                orientation="h",
                title="Distribución por categoría",
            )

            fig.update_layout(
                showlegend=False,
                height=260,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# -----------------------------------------------------------------------------
# TAB 2 - Batch CSV
# -----------------------------------------------------------------------------

def tab_batch(
    sent_pipe,
    emo_pipe,
    cat_pipe,
) -> None:

    st.subheader(
        "Batch CSV Analysis"
    )

    st.caption(
        "Upload a CSV with a column named **text** "
        "— get aggregate dashboards."
    )

    upload = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    use_demo = st.checkbox(
        "Use demo data instead",
        value=upload is None,
    )

    df_in: pd.DataFrame | None = None

    # -------------------------------------------------------------------------
    # Uploaded CSV
    # -------------------------------------------------------------------------

    if upload is not None:

        try:

            df_in = pd.read_csv(
                upload
            )

        except Exception as e:

            st.error(
                f"Error reading CSV: {e}"
            )

            return

    # -------------------------------------------------------------------------
    # Demo data
    # -------------------------------------------------------------------------

    elif use_demo:

        df_in = pd.DataFrame(
            {
                "text": [
                    "The food was absolutely delicious, best meal of my year!",
                    "Service was painfully slow and the staff seemed uninterested.",
                    "Decent value for money, nothing extraordinary though.",
                    "I am furious — they cancelled my order without any notice.",
                    "Pure joy seeing my package arrive a day early.",
                    "Ce produit est incroyable, je le recommande vivement.",
                    "Estoy muy decepcionado con la calidad del servicio.",
                    "Das Hotel war sehr schön und das Personal freundlich.",
                    "Honestly, I'm scared to try this brand again.",
                    "Surprisingly good for the price point — pleasantly impressed.",
                ]
            }
        )

    if df_in is None:
        return

    # -------------------------------------------------------------------------
    # Validate CSV
    # -------------------------------------------------------------------------

    if "text" not in df_in.columns:

        st.error(
            "CSV must contain a column named `text`."
        )

        return

    st.write(
        f"**Rows:** {len(df_in)}"
    )

    st.dataframe(
        df_in.head(10),
        use_container_width=True,
    )

    # -------------------------------------------------------------------------
    # Run analysis
    # -------------------------------------------------------------------------

    if st.button(
        "Run Batch Analysis",
        type="primary",
        key="batch_btn",
    ):

        df_out = analyse_batch(
            df_in["text"].tolist(),
            sent_pipe,
            emo_pipe,
            cat_pipe,
        )

        if df_out.empty:

            st.warning(
                "No valid text rows to analyse."
            )

            return

        st.success(
            f"Analysed {len(df_out)} rows."
        )

        # ---------------------------------------------------------------------
        # KPI row
        # ---------------------------------------------------------------------

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Total",
            len(df_out),
        )

        c2.metric(
            "Positive",
            int(
                (
                    df_out["sentiment"] == "positive"
                ).sum()
            ),
        )

        c3.metric(
            "Negative",
            int(
                (
                    df_out["sentiment"] == "negative"
                ).sum()
            ),
        )

        c4.metric(
            "Avg Confidence",
            f'{df_out["sentiment_confidence"].mean() * 100:.1f}%',
        )

        c5.metric(
            "Categorías",
            df_out["category"].nunique(),
        )

        # ---------------------------------------------------------------------
        # Sentiment + Emotion
        # ---------------------------------------------------------------------

        cc1, cc2 = st.columns(2)

        with cc1:

            sent_counts = (
                df_out["sentiment"]
                .value_counts()
                .reset_index()
            )

            sent_counts.columns = [
                "sentiment",
                "count",
            ]

            fig = px.pie(
                sent_counts,
                values="count",
                names="sentiment",
                color="sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                title="Sentiment distribution",
                hole=0.45,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with cc2:

            emo_counts = (
                df_out["emotion"]
                .value_counts()
                .reset_index()
            )

            emo_counts.columns = [
                "emotion",
                "count",
            ]

            fig = px.bar(
                emo_counts,
                x="emotion",
                y="count",
                color="emotion",
                color_discrete_map=EMOTION_COLORS,
                title="Emotion distribution",
            )

            fig.update_layout(
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        # ---------------------------------------------------------------------
        # Category distribution
        # ---------------------------------------------------------------------

        st.subheader(
            "📊 Análisis por categoría"
        )

        cat_counts = (
            df_out["category"]
            .value_counts()
            .reset_index()
        )

        cat_counts.columns = [
            "category",
            "count",
        ]

        fig = px.bar(
            cat_counts,
            x="category",
            y="count",
            text="count",
            title="Cantidad de comentarios por categoría",
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # ---------------------------------------------------------------------
        # Category percentage
        # ---------------------------------------------------------------------

        cat_percentage = (
            df_out["category"]
            .value_counts(normalize=True)
            .mul(100)
            .round(1)
            .reset_index()
        )

        cat_percentage.columns = [
            "category",
            "percentage",
        ]

        fig = px.pie(
            cat_percentage,
            values="percentage",
            names="category",
            title="Participación por categoría",
            hole=0.45,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # ---------------------------------------------------------------------
        # Sentiment confidence
        # ---------------------------------------------------------------------

        fig = px.histogram(
            df_out,
            x="sentiment_confidence",
            nbins=20,
            title="Sentiment confidence histogram",
            color_discrete_sequence=["#3498db"],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # ---------------------------------------------------------------------
        # Sentiment × Emotion
        # ---------------------------------------------------------------------

        cross = pd.crosstab(
            df_out["sentiment"],
            df_out["emotion"],
        )

        fig = px.imshow(
            cross,
            text_auto=True,
            aspect="auto",
            title="Sentiment × Emotion co-occurrence",
            color_continuous_scale="Blues",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # ---------------------------------------------------------------------
        # Category × Sentiment
        # ---------------------------------------------------------------------

        st.subheader(
            "📌 Categoría × Sentiment"
        )

        cat_sent = pd.crosstab(
            df_out["category"],
            df_out["sentiment"],
        )

        fig = px.bar(
            cat_sent,
            barmode="group",
            title="Sentimiento por categoría",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # ---------------------------------------------------------------------
        # Negative percentage by category
        # ---------------------------------------------------------------------

        category_negative = (
            df_out.assign(
                negative=(
                    df_out["sentiment"] == "negative"
                ).astype(int)
            )
            .groupby("category")["negative"]
            .mean()
            .mul(100)
            .round(1)
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        category_negative.columns = [
            "category",
            "negative_percentage",
        ]

        st.subheader(
            "🚨 Porcentaje negativo por categoría"
        )

        fig = px.bar(
            category_negative,
            x="category",
            y="negative_percentage",
            text="negative_percentage",
            title="Porcentaje de comentarios negativos por categoría",
            labels={
                "negative_percentage": "Negativos (%)",
                "category": "Categoría",
            },
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # ---------------------------------------------------------------------
        # Detailed results
        # ---------------------------------------------------------------------

        st.subheader(
            "Detailed results"
        )

        st.dataframe(
            df_out,
            use_container_width=True,
        )

        # ---------------------------------------------------------------------
        # Download CSV
        # ---------------------------------------------------------------------

        buf = io.StringIO()

        df_out.to_csv(
            buf,
            index=False,
        )

        st.download_button(
            "📥 Download results CSV",
            buf.getvalue(),
            file_name="sentiment_emotion_category_results.csv",
            mime="text/csv",
        )


# -----------------------------------------------------------------------------
# TAB 3 - Compare
# -----------------------------------------------------------------------------

def tab_compare(
    sent_pipe,
    emo_pipe,
    cat_pipe,
) -> None:

    st.subheader(
        "Side-by-Side Comparison"
    )

    st.caption(
        "Enter up to 5 short texts to compare "
        "sentiment, emotion and category."
    )

    default_texts = [
        "I love this product, it's amazing!",
        "Worst experience I've ever had.",
        "It was okay, nothing special.",
    ]

    cols = st.columns(2)

    inputs: List[str] = []

    for i in range(5):

        with cols[i % 2]:

            txt = st.text_input(
                f"Text {i + 1}",
                value=(
                    default_texts[i]
                    if i < len(default_texts)
                    else ""
                ),
                key=f"cmp_{i}",
            )

            if txt.strip():
                inputs.append(txt)

    if (
        st.button(
            "Compare",
            type="primary",
            key="cmp_btn",
        )
        and inputs
    ):

        df = analyse_batch(
            inputs,
            sent_pipe,
            emo_pipe,
            cat_pipe,
        )

        st.dataframe(
            df,
            use_container_width=True,
        )

        fig = px.bar(
            df,
            x=df.index,
            y="sentiment_confidence",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            hover_data=[
                "text",
                "emotion",
                "category",
                "category_confidence",
            ],
            title="Sentiment confidence per input",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # Category comparison

        fig = px.bar(
            df,
            x=df.index,
            y="category_confidence",
            color="category",
            hover_data=[
                "text",
                "sentiment",
                "emotion",
            ],
            title="Category confidence per input",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:

    st.title(
        "💬 Multi-lingual Sentiment & Emotion Dashboard"
    )

    st.caption(
        "Powered by HuggingFace Transformers · "
        "XLM-RoBERTa + DistilRoBERTa + Zero-Shot Classification"
    )

    render_sidebar()

    # Load all three models
    sent_pipe = load_sentiment()
    emo_pipe = load_emotion()
    cat_pipe = load_category()

    # Tabs
    t1, t2, t3 = st.tabs(
        [
            "📝 Single Text",
            "📊 Batch CSV",
            "⚖️ Compare",
        ]
    )

    with t1:

        tab_single(
            sent_pipe,
            emo_pipe,
            cat_pipe,
        )

    with t2:

        tab_batch(
            sent_pipe,
            emo_pipe,
            cat_pipe,
        )

    with t3:

        tab_compare(
            sent_pipe,
            emo_pipe,
            cat_pipe,
        )


# -----------------------------------------------------------------------------
# Run application
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
```
