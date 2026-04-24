import streamlit as st
from PIL import Image

from mdif.inference import load_models, analyze_image

st.set_page_config(page_title="MDIF", page_icon=":guardsman:", layout="wide")

st.title("MDIF: Multi-Domain Inconsistency Framework")
st.caption("A Multi-Domain Inconsistency Framework proposed for our Image Processing course (CS F311).")

@st.cache_resource(show_spinner="Loading models", show_time=True)
def get_models():
    return load_models()


models = get_models()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    image = Image.open(uploaded)

    col_img, col_results = st.columns([1, 1], gap="large")

    with col_img:
        st.html(
            """
            <style>
                div[data-testid='stFullScreenFrame'] { display: flex; justify-content: center; }
                div[data-testid='stImage'] img { max-height: 40vh; width: auto !important; }
            </style>
            """,
            width="stretch"
        )
        st.image(image, caption=uploaded.name)

    with col_results:
        with st.spinner("Analysing..."):
            label, confidence, class_probabilities = analyze_image(image, models=models)

        verdict_color = {
            "Authentic Photograph": "green",
            "Fully AI-Generated": "red",
            "Partially AI-Inpainted": "yellow",
        }.get(label, "gray")

        st.metric("Prediction", f":{verdict_color}[{label}]")
        st.metric("Confidence", f"{confidence * 100:.1f}%")

        st.subheader("All Scores")
        for class_name, prob in class_probabilities.items():
            st.progress(prob, text=f"{class_name} | {prob * 100:.1f}%")
