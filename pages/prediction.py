import pandas as pd
import streamlit as st
import joblib
from datetime import datetime

def show():
    # --------------------------------------------------
    # Load saved models and preprocessing objects
    # --------------------------------------------------
    regression_model = joblib.load("G:/Project/Realestate_project/Scripts/best_regression_model.pkl")
    regression_preprocessor = joblib.load("G:/Project/Realestate_project/Scripts/regression_preprocessor.pkl")
    classification_model = joblib.load("G:/Project/Realestate_project/Scripts/best_classification_model.pkl")
    classification_preprocessor = joblib.load("G:/Project/Realestate_project/Scripts/classification_preprocessor.pkl")
    # --------------------------------------------------
    # Streamlit page
    # --------------------------------------------------
    st.title("🏠 Property Price and Investment Prediction")
    st.write(
        "Enter the property details below to estimate the future property "
        "price and investment suitability.")
    # --------------------------------------------------
    # User inputs
    # --------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        bhk = st.number_input("BHK",min_value=1,max_value=10,value=2,step=1)
        size = st.number_input("Size in SqFt",min_value=100.0,value=1000.0,step=50.0)
        current_price = st.number_input("Current Property Price (₹)",min_value=100000.0,value=5000000.0,step=100000.0)
        year_built = st.number_input("Year Built",min_value=1900,max_value=datetime.now().year,value=2020,step=1)
        floor_no = st.number_input("Floor Number",min_value=0,value=1,step=1)
        nearby_schools = st.number_input("Nearby Schools",min_value=0,value=2,step=1)
    with col2:
        nearby_hospitals = st.number_input("Nearby Hospitals",min_value=0,value=1,step=1)
        locality = st.text_input("Locality",value="Velachery")
        property_type = st.selectbox("Property Type",["Apartment", "Villa", "Independent House", "Plot"])
        furnished_status = st.selectbox("Furnished Status",["Furnished", "Semi-Furnished", "Unfurnished"])
        security = st.selectbox("Security",["Yes", "No"])
    # --------------------------------------------------
    # Prediction button
    # --------------------------------------------------
    if st.button("Predict"):
        current_year = datetime.now().year
        age = current_year - int(year_built)
        price_per_sqft = current_price / size
        # Default values
        total_floors = 5
        public_transport = "Good"
        parking = "Available"
        facing = "East"
        owner_type = "Builder"
        availability = "Ready_to_Move"
        # Default amenities
        garden = 0
        gym = 0
        playground = 0
        clubhouse = 0
        pool = 0
        # Default location
        state = "Tamil Nadu"
        city = "Chennai"
        # --------------------------------------------------
        # Regression input
        # --------------------------------------------------
        regression_input = pd.DataFrame([{
            "BHK": bhk,
            "Size_in_SqFt": size,
            "Year_Built": year_built,
            "Floor_No": floor_no,
            "Total_Floors": total_floors,
            "Age_of_Property": age,
            "Nearby_Schools": nearby_schools,
            "Nearby_Hospitals": nearby_hospitals,
            "Amenity_Garden": garden,
            "Amenity_Gym": gym,
            "Amenity_Playground": playground,
            "Amenity_Clubhouse": clubhouse,
            "Amenity_Pool": pool,
            "State": state,
            "City": city,
            "Locality": locality,
            "Property_Type": property_type,
            "Furnished_Status": furnished_status,
            "Public_Transport_Accessibility": public_transport,
            "Parking_Space": parking,
            "Security": security,
            "Facing": facing,
            "Owner_Type": owner_type,
            "Availability_Status": availability
        }])
        # --------------------------------------------------
        # Classification input
        # --------------------------------------------------
        classification_input = regression_input.copy()
        classification_input["Price"] = current_price
        classification_input["Price_per_SqFt"] = price_per_sqft
        # --------------------------------------------------
        # Regression prediction
        # --------------------------------------------------
        regression_processed = regression_preprocessor.transform(regression_input)
        predicted_price = regression_model.predict(regression_processed)[0]
        # --------------------------------------------------
        # Classification prediction
        # --------------------------------------------------
        classification_processed = classification_preprocessor.transform(classification_input)
        investment_prediction = classification_model.predict(classification_processed)[0]
        # --------------------------------------------------
        # Display results
        # --------------------------------------------------
        st.subheader("Prediction Results")
        st.success(f"Estimated Future Price after 5 Years: "f"₹{predicted_price:,.2f}")
        if investment_prediction == 1:
            st.success("Good Investment: Yes ✅")
        else:
            st.warning("Good Investment: No ❌")