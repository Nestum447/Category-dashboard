import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from transformers import pipeline


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Social Media Sentiment & Topic Analysis",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# MODELOS
# ============================================================

SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

TOPIC_MODEL = "facebook/bart-large-mnli"


# ============================================================
# CATEGORÍAS
# ============================================================

TOPIC_LABELS = [
    "precio o costo",
    "calidad del producto",
    "tiempo o problemas de entrega",
    "servicio al cliente",
    "atención del personal",
    "características del producto",
    "disponibilidad del producto",
    "intención de compra",
    "ubicación de la empresa",
    "garantía o devolución",
    "otro"
]


# ============================================================
# CARGAR MODELOS
# ============================================================

@st.cache_resource(show_spinner="Cargando modelo de sentimiento...")
def load_sentiment():

    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        top_k=None
    )


@st.cache_resource(show_spinner="Cargando modelo de emociones...")
def load_emotion():

    return pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        top_k=None
    )


@st.cache_resource(show_spinner="Cargando modelo de categorías...")
def load_topic_model():

    return pipeline(
        "zero-shot-classification",
        model=TOPIC_MODEL
    )


# ============================================================
# SENTIMIENTO
# ============================================================

def analyse_sentiment(text, sent_pipe):

    result = sent_pipe(text[:512])

    if isinstance(result, list):

        # Cuando top_k=None devuelve lista de resultados
        if len(result) > 0 and isinstance(result[0], list):
            result = result[0]

        best = max(
            result,
            key=lambda x: x["score"]
        )

    else:

        best = result

    label = best["label"]
    confidence = float(best["score"])

    # Normalización de etiquetas
    label_upper = label.upper()

    if label_upper in ["LABEL_2", "POSITIVE", "POS"]:
        sentiment = "positive"

    elif label_upper in ["LABEL_0", "NEGATIVE", "NEG"]:
        sentiment = "negative"

    else:
        sentiment = "neutral"

    return sentiment, confidence, result


# ============================================================
# EMOCIÓN
# ============================================================

def analyse_emotion(text, emo_pipe):

    result = emo_pipe(text[:512])

    if isinstance(result, list):

        if len(result) > 0 and isinstance(result[0], list):
            result = result[0]

        best = max(
            result,
            key=lambda x: x["score"]
        )

    else:

        best = result

    emotion = best["label"]
    confidence = float(best["score"])

    return emotion, confidence, result


# ============================================================
# CATEGORIZACIÓN
# ============================================================

def analyse_topics(
    text,
    topic_pipe,
    threshold=0.50
):

    result = topic_pipe(
        text[:512],
        candidate_labels=TOPIC_LABELS,
        multi_label=True
    )

    topics = []

    for label, score in zip(
        result["labels"],
        result["scores"]
    ):

        if float(score) >= threshold:

            topics.append({
                "topic": label,
                "confidence": round(
                    float(score),
                    4
                )
            })

    return topics


# ============================================================
# ANÁLISIS COMPLETO DE UN COMENTARIO
# ============================================================

def analyse_one(
    text,
    sent_pipe,
    emo_pipe,
    topic_pipe=None,
    topic_threshold=0.50
):

    text = str(text).strip()

    if not text:

        return {
            "text": "",
            "sentiment": "neutral",
            "sentiment_confidence": 0,
            "emotion": "unknown",
            "emotion_confidence": 0,
            "topics": "",
            "topic_details": ""
        }

    # ----------------------------
    # SENTIMIENTO
    # ----------------------------

    sentiment, sentiment_confidence, sent_dist = (
        analyse_sentiment(
            text,
            sent_pipe
        )
    )

    # ----------------------------
    # EMOCIÓN
    # ----------------------------

    emotion, emotion_confidence, emo_dist = (
        analyse_emotion(
            text,
            emo_pipe
        )
    )

    # ----------------------------
    # CATEGORÍAS
    # ----------------------------

    topics = []

    if topic_pipe is not None:

        topics = analyse_topics(
            text,
            topic_pipe,
            topic_threshold
        )

    topic_names = ", ".join(
        [x["topic"] for x in topics]
    )

    topic_details = ", ".join(
        [
            f'{x["topic"]}: {x["confidence"]:.2f}'
            for x in topics
        ]
    )

    return {

        "text": text,

        "sentiment": sentiment,

        "sentiment_confidence": round(
            sentiment_confidence,
            4
        ),

        "emotion": emotion,

        "emotion_confidence": round(
            emotion_confidence,
            4
        ),

        "topics": topic_names,

        "topic_details": topic_details,

        "_sent_dist": sent_dist,

        "_emo_dist": emo_dist
    }


# ============================================================
# ANÁLISIS MASIVO
# ============================================================

def analyse_batch(
    texts,
    sent_pipe,
    emo_pipe,
    topic_pipe=None,
    topic_threshold=0.50
):

    results = []

    progress = st.progress(0)

    total = len(texts)

    for i, text in enumerate(texts):

        try:

            result = analyse_one(
                text,
                sent_pipe,
                emo_pipe,
                topic_pipe,
                topic_threshold
            )

            results.append(result)

        except Exception as e:

            results.append({

                "text": text,

                "sentiment": "error",

                "sentiment_confidence": 0,

                "emotion": "error",

                "emotion_confidence": 0,

                "topics": "error",

                "topic_details": str(e)
            })

        progress.progress(
            int(((i + 1) / total) * 100)
        )

    progress.empty()

    return pd.DataFrame(results)


# ============================================================
# TAB SINGLE TEXT
# ============================================================

def tab_single(
    sent_pipe,
    emo_pipe,
    topic_pipe
):

    st.header("🔎 Analyze a comment")

    text = st.text_area(
        "Write a Facebook comment:",
        height=150,
        placeholder=(
            "Example: "
            "The product is excellent but the price "
            "is too high and delivery was very slow."
        )
    )

    topic_threshold = st.slider(
        "Category confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    if st.button(
        "Analyze comment",
        type="primary"
    ):

        if not text.strip():

            st.warning(
                "Please enter a comment."
            )

            return

        result = analyse_one(
            text,
            sent_pipe,
            emo_pipe,
            topic_pipe,
            topic_threshold
        )

        # ====================================================
        # RESULTADOS PRINCIPALES
        # ====================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Sentiment",
                result["sentiment"].upper()
            )

            st.caption(
                f'Confidence: '
                f'{result["sentiment_confidence"]:.2%}'
            )

        with col2:

            st.metric(
                "Emotion",
                result["emotion"].upper()
            )

            st.caption(
                f'Confidence: '
                f'{result["emotion_confidence"]:.2%}'
            )

        with col3:

            st.metric(
                "Categories",
                len(
                    result["topics"].split(", ")
                )
                if result["topics"]
                else 0
            )

        # ====================================================
        # CATEGORÍAS
        # ====================================================

        st.subheader("🏷️ Detected categories")

        if result["topics"]:

            topics = result["topics"].split(", ")

            for topic in topics:

                st.success(topic)

        else:

            st.info(
                "No category exceeded the selected threshold."
            )

        # ====================================================
        # DETALLE DE CATEGORÍAS
        # ====================================================

        if result["topic_details"]:

            st.subheader(
                "Category confidence"
            )

            st.write(
                result["topic_details"]
            )


# ============================================================
# TAB BATCH CSV
# ============================================================

def tab_batch(
    sent_pipe,
    emo_pipe,
    topic_pipe
):

    st.header("📂 Analyze Facebook comments in bulk")

    st.write(
        "Upload a CSV file containing a column called `text`."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    topic_threshold = st.slider(
        "Category confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        key="batch_threshold"
    )

    if uploaded_file is None:

        st.info(
            "Example CSV structure:"
        )

        example = pd.DataFrame({

            "text": [

                "Excellent product, very good quality",

                "The price is too expensive",

                "Delivery was very slow",

                "I want to buy this product",

                "Nobody answered my message"

            ]
        })

        st.dataframe(
            example,
            use_container_width=True
        )

        return

    # ========================================================
    # READ CSV
    # ========================================================

    try:

        df = pd.read_csv(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Error reading CSV: {e}"
        )

        return

    # ========================================================
    # VALIDATE COLUMN
    # ========================================================

    if "text" not in df.columns:

        st.error(
            "The CSV must contain a column called 'text'."
        )

        st.write(
            "Columns found:"
        )

        st.write(
            list(df.columns)
        )

        return

    # ========================================================
    # PREVIEW
    # ========================================================

    st.subheader("📄 Input data")

    st.write(
        f"Comments: **{len(df):,}**"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    # ========================================================
    # LIMIT
    # ========================================================

    max_comments = st.number_input(
        "Maximum comments to analyze",
        min_value=1,
        max_value=len(df),
        value=min(100, len(df)),
        step=10
    )

    if st.button(
        "🚀 Analyze comments",
        type="primary"
    ):

        df_process = df.head(
            int(max_comments)
        ).copy()

        with st.spinner(
            "Analyzing comments..."
        ):

            df_out = analyse_batch(
                df_process["text"].tolist(),
                sent_pipe,
                emo_pipe,
                topic_pipe,
                topic_threshold
            )

        st.success(
            f"Analysis completed: "
            f"{len(df_out):,} comments."
        )

        # ====================================================
        # RESULTS
        # ====================================================

        st.subheader(
            "📊 Analysis results"
        )

        st.dataframe(
            df_out[
                [
                    "text",
                    "sentiment",
                    "sentiment_confidence",
                    "emotion",
                    "emotion_confidence",
                    "topics",
                    "topic_details"
                ]
            ],
            use_container_width=True
        )

        # ====================================================
        # KPIs
        # ====================================================

        st.subheader(
            "📌 Key metrics"
        )

        col1, col2, col3, col4 = st.columns(4)

        total = len(df_out)

        positive = (
            df_out["sentiment"]
            .eq("positive")
            .sum()
        )

        negative = (
            df_out["sentiment"]
            .eq("negative")
            .sum()
        )

        neutral = (
            df_out["sentiment"]
            .eq("neutral")
            .sum()
        )

        with col1:

            st.metric(
                "Total comments",
                f"{total:,}"
            )

        with col2:

            st.metric(
                "Positive",
                f"{positive:,}"
            )

        with col3:

            st.metric(
                "Negative",
                f"{negative:,}"
            )

        with col4:

            st.metric(
                "Neutral",
                f"{neutral:,}"
            )

        # ====================================================
        # SENTIMENT CHART
        # ====================================================

        st.subheader(
            "😊 Sentiment distribution"
        )

        sentiment_counts = (
            df_out["sentiment"]
            .value_counts()
        )

        fig1, ax1 = plt.subplots()

        sentiment_counts.plot(
            kind="bar",
            ax=ax1
        )

        ax1.set_xlabel(
            "Sentiment"
        )

        ax1.set_ylabel(
            "Comments"
        )

        ax1.set_title(
            "Sentiment Distribution"
        )

        plt.xticks(
            rotation=0
        )

        st.pyplot(
            fig1
        )

        # ====================================================
        # EMOTION
        # ====================================================

        st.subheader(
            "🎭 Emotion distribution"
        )

        emotion_counts = (
            df_out["emotion"]
            .value_counts()
        )

        fig2, ax2 = plt.subplots()

        emotion_counts.plot(
            kind="bar",
            ax=ax2
        )

        ax2.set_xlabel(
            "Emotion"
        )

        ax2.set_ylabel(
            "Comments"
        )

        ax2.set_title(
            "Emotion Distribution"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        st.pyplot(
            fig2
        )

        # ====================================================
        # TOPICS
        # ====================================================

        st.subheader(
            "🏷️ Main comment categories"
        )

        topic_counts = (
            df_out["topics"]
            .fillna("")
            .str.split(", ")
            .explode()
        )

        topic_counts = topic_counts[
            topic_counts != ""
        ]

        topic_counts = (
            topic_counts
            .value_counts()
        )

        if len(topic_counts) > 0:

            fig3, ax3 = plt.subplots()

            topic_counts.sort_values().plot(
                kind="barh",
                ax=ax3
            )

            ax3.set_xlabel(
                "Number of comments"
            )

            ax3.set_ylabel(
                "Category"
            )

            ax3.set_title(
                "Comment Categories"
            )

            st.pyplot(
                fig3
            )

        else:

            st.info(
                "No categories detected."
            )

        # ====================================================
        # SENTIMENT BY TOPIC
        # ====================================================

        st.subheader(
            "📈 Sentiment by category"
        )

        rows = []

        for _, row in df_out.iterrows():

            topics = str(
                row["topics"]
            )

            if topics:

                for topic in topics.split(", "):

                    if topic:

                        rows.append({

                            "topic": topic,

                            "sentiment":
                                row["sentiment"]

                        })

        if rows:

            topic_sentiment = pd.DataFrame(
                rows
            )

            pivot = pd.crosstab(
                topic_sentiment["topic"],
                topic_sentiment["sentiment"]
            )

            st.dataframe(
                pivot,
                use_container_width=True
            )

            fig4, ax4 = plt.subplots()

            pivot.plot(
                kind="bar",
                ax=ax4
            )

            ax4.set_xlabel(
                "Category"
            )

            ax4.set_ylabel(
                "Number of comments"
            )

            ax4.set_title(
                "Sentiment by Category"
            )

            plt.xticks(
                rotation=45,
                ha="right"
            )

            st.pyplot(
                fig4
            )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        st.subheader(
            "📥 Download results"
        )

        download_df = df_out[
            [
                "text",
                "sentiment",
                "sentiment_confidence",
                "emotion",
                "emotion_confidence",
                "topics",
                "topic_details"
            ]
        ].copy()

        csv = download_df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            label="⬇️ Download analyzed CSV",
            data=csv,
            file_name="facebook_comments_analysis.csv",
            mime="text/csv"
        )


# ============================================================
# TAB COMPARE
# ============================================================

def tab_compare(
    sent_pipe,
    emo_pipe
):

    st.header(
        "⚖️ Compare comments"
    )

    text1 = st.text_area(
        "Comment 1",
        height=100
    )

    text2 = st.text_area(
        "Comment 2",
        height=100
    )

    if st.button(
        "Compare"
    ):

        if not text1.strip() or not text2.strip():

            st.warning(
                "Enter both comments."
            )

            return

        r1 = analyse_one(
            text1,
            sent_pipe,
            emo_pipe
        )

        r2 = analyse_one(
            text2,
            sent_pipe,
            emo_pipe
        )

        comparison = pd.DataFrame({

            "Comment": [
                "Comment 1",
                "Comment 2"
            ],

            "Sentiment": [
                r1["sentiment"],
                r2["sentiment"]
            ],

            "Confidence": [
                r1["sentiment_confidence"],
                r2["sentiment_confidence"]
            ],

            "Emotion": [
                r1["emotion"],
                r2["emotion"]
            ]

        })

        st.dataframe(
            comparison,
            use_container_width=True
        )


# ============================================================
# MAIN
# ============================================================

def main():

    st.title(
        "📊 Social Media Sentiment & Category Analysis"
    )

    st.markdown(
        """
        Analyze Facebook comments using AI.

        **The application detects:**

        - 😊 Sentiment
        - 🎭 Emotion
        - 🏷️ Comment categories
        - 📊 Category frequency
        - 📈 Sentiment by category
        """
    )

    # ========================================================
    # LOAD MODELS
    # ========================================================

    try:

        sent_pipe = load_sentiment()

        emo_pipe = load_emotion()

        topic_pipe = load_topic_model()

    except Exception as e:

        st.error(
            "Error loading AI models."
        )

        st.exception(e)

        return

    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "🔎 Single Text",
            "📂 Batch CSV",
            "⚖️ Compare"
        ]
    )

    with tab1:

        tab_single(
            sent_pipe,
            emo_pipe,
            topic_pipe
        )

    with tab2:

        tab_batch(
            sent_pipe,
            emo_pipe,
            topic_pipe
        )

    with tab3:

        tab_compare(
            sent_pipe,
            emo_pipe
        )


# ============================================================
# RUN 
# ============================================================

if __name__ == "__main__":

    main()
