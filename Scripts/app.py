import streamlit as st
from pages import home, analysis, prediction

st.set_page_config(
    page_title="Real Estate Analytics",
    page_icon="🏠",
    layout="wide"
)

st.sidebar.title("🏠 Real Estate Analytics")

page = st.sidebar.radio(
    "Navigation",
    (
        "🏠 Introduction",
        "📊 Analysis / EDA",
        "🔮 Prediction"
    )
)

if page == "🏠 Introduction":
    home.show()

elif page == "📊 Analysis / EDA":
    analysis.show()

elif page == "🔮 Prediction":
    prediction.show()