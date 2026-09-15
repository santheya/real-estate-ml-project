import streamlit as st


def show():
    st.title("📘 Project Introduction")

    st.header("Real Estate Analytics and Prediction System")

    st.write(
        """
        This project analyzes real-estate property data and uses machine
        learning models to estimate future property prices and identify
        potentially good investment properties.
        """
    )

    st.header("Project Objectives")

    objectives = [
        "Clean and preprocess real-estate data.",
        "Perform exploratory data analysis.",
        "Analyze property prices, sizes, locations, and amenities.",
        "Predict property prices after five years.",
        "Classify properties as good or not good investments.",
        "Develop an interactive Streamlit application."
    ]

    for objective in objectives:
        st.write(f"• {objective}")