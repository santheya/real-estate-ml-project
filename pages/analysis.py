import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    return pd.read_csv("G:/Project/Realestate_project/Scripts/property_data_updated.csv")

def show():
    st.title("📊 Analysis / Exploratory Data Analysis")
    df = load_data()
    heading = st.selectbox(
    "Select EDA Heading",[
        "Price & Size Analysis",
        "Location-based Analysis",
        "Feature Relationship & Correlation",
        "Investment / Amenities / Ownership Analysis"])
    if heading == "Price & Size Analysis":
        st.write(
            "This section analyzes property prices, property sizes, "
            "price per square foot, relationships, and outliers.")
        # --------------------------------------------------
        # Row 1 - Questions 1 and 2
        # --------------------------------------------------
        col1, col2 = st.columns(2)
        # Question 1
        with col1:
            st.subheader("1. What is the distribution of property prices?")
            price_data = df[["Price"]].dropna()
            fig_price = px.histogram(price_data,x="Price",nbins=50,title="Distribution of Property Prices",labels={"Price": "Property Price (INR)","count": "Number of Properties"})
            fig_price.update_layout(height=450,template="plotly_white")
            st.plotly_chart(fig_price,use_container_width=True)
        # Question 2
        with col2:
            st.subheader("2. What is the distribution of property sizes?")
            size_data = df[["Size_in_SqFt"]].dropna()
            fig_size = px.histogram(size_data,x="Size_in_SqFt",nbins=50,title="Distribution of Property Sizes",labels={"Size_in_SqFt": "Property Size (Sq Ft)","count": "Number of Properties"})
            fig_size.update_layout(height=450,template="plotly_white")
            st.plotly_chart(fig_size,use_container_width=True)
        # --------------------------------------------------
        # Row 2 - Questions 3 and 4
        # --------------------------------------------------
        col3, col4 = st.columns(2)
        # Question 3
        with col3:
            st.subheader("3. How does the price per sq ft vary by property type?" )
            price_type_data = df[["Property_Type","Price_per_SqFt"]].dropna()
            fig_price_type = px.box(
                price_type_data,x="Property_Type",y="Price_per_SqFt",color="Property_Type",title="Price per Sq Ft by Property Type",labels={"Property_Type": "Property Type","Price_per_SqFt": "Price per Sq Ft (INR)"})
            fig_price_type.update_layout(height=450,showlegend=False,template="plotly_white")
            st.plotly_chart(fig_price_type,use_container_width=True)
        # Question 4
        with col4:
            st.subheader("4. Is there a relationship between property size and price?")
            relationship_data = df[["Size_in_SqFt","Price"]].dropna()
            fig_relationship = px.scatter(relationship_data,x="Size_in_SqFt",y="Price",trendline="ols",opacity=0.5,title="Relationship Between Property Size and Price",labels={"Size_in_SqFt": "Property Size (Sq Ft)","Price": "Property Price (INR)"})
            fig_relationship.update_layout(height=450,template="plotly_white")
            st.plotly_chart(fig_relationship,use_container_width=True)
            correlation = relationship_data["Size_in_SqFt"].corr(relationship_data["Price"])
        # --------------------------------------------------
        # Row 3 - Question 5 in the middle
        # --------------------------------------------------
        left_space, middle_col, right_space = st.columns([1, 2, 1])
        with middle_col:
            st.subheader("5. Are there any outliers in price per sq ft or property size?")
            outlier_col1, outlier_col2 = st.columns(2)
            # Price per Sq Ft outliers
            with outlier_col1:
                price_outlier_data = df[["Price_per_SqFt"]].dropna()
                fig_price_outliers = px.box(price_outlier_data,y="Price_per_SqFt",title="Price per Sq Ft Outliers",labels={"Price_per_SqFt": "Price per Sq Ft (INR)"})
                fig_price_outliers.update_layout(height=400,template="plotly_white")
                st.plotly_chart(fig_price_outliers,use_container_width=True)
            # Property size outliers
            with outlier_col2:
                size_outlier_data = df[["Size_in_SqFt"]].dropna()
                fig_size_outliers = px.box(size_outlier_data,y="Size_in_SqFt",title="Property Size Outliers",labels={"Size_in_SqFt": "Property Size (Sq Ft)"})
                fig_size_outliers.update_layout(height=400,template="plotly_white")
                st.plotly_chart(fig_size_outliers,use_container_width=True)
    elif heading == "Location-based Analysis":
        st.header("📍 Location-based Analysis")
        # --------------------------------------------------
        # Question 6 and Question 7
        # --------------------------------------------------
        col1, col2 = st.columns(2)
        # Question 6
        with col1:
            st.subheader("6. What is the average price per sq ft by state?")
            state_data = (df.groupby("State", as_index=False)["Price_per_SqFt"].mean().sort_values("Price_per_SqFt", ascending=False))
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=state_data,x="State",y="Price_per_SqFt",ax=ax)
            ax.set_title("Average Price per Sq Ft by State")
            ax.set_xlabel("State")
            ax.set_ylabel("Average Price per Sq Ft (INR)")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        # Question 7
        with col2:
            st.subheader("7. What is the average property price by city?")
            city_data = (df.groupby("City", as_index=False)["Price"].mean().sort_values("Price", ascending=False))
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=city_data,x="City",y="Price",ax=ax)
            ax.set_title("Average Property Price by City")
            ax.set_xlabel("City")
            ax.set_ylabel("Average Property Price (INR)")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        # --------------------------------------------------
        # Question 8 and Question 9
        # --------------------------------------------------
        col3, col4 = st.columns(2)
        # Question 8
        with col3:
            st.subheader("8. What is the median age of properties by locality?")
            locality_age_data = (df.groupby("Locality", as_index=False)["Age_of_Property"].median().sort_values("Age_of_Property", ascending=False))
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=locality_age_data,x="Locality",y="Age_of_Property",ax=ax)
            ax.set_title("Median Property Age by Locality")
            ax.set_xlabel("Locality")
            ax.set_ylabel("Median Age of Property (Years)")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        # Question 9
        with col4:
            st.subheader("9. How is BHK distributed across cities?")
            bhk_city_data = (df.groupby(["City", "BHK"]).size().reset_index(name="Property_Count"))
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=bhk_city_data,x="City",y="Property_Count",hue="BHK",ax=ax)
            ax.set_title("BHK Distribution Across Cities")
            ax.set_xlabel("City")
            ax.set_ylabel("Number of Properties")
            ax.tick_params(axis="x", rotation=45)
            ax.legend(title="BHK",bbox_to_anchor=(1.05, 1),loc="upper left")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        # --------------------------------------------------
        # Question 10 - Centered
        # --------------------------------------------------
        left_space, middle_col, right_space = st.columns([1, 2, 1])
        with middle_col:
            st.subheader("10. What are the price trends for the top 5 most expensive localities?")
            top_localities = (df.groupby("Locality")["Price"].mean().nlargest(5).index)
            top_locality_data = df[df["Locality"].isin(top_localities)].copy()
            top_locality_data["Year_Built"] = pd.to_numeric(top_locality_data["Year_Built"],errors="coerce")
            locality_trend_data = (top_locality_data.groupby(["Year_Built", "Locality"], as_index=False)["Price"].mean().dropna().sort_values("Year_Built"))
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=locality_trend_data,x="Year_Built",y="Price",hue="Locality",marker="o",ax=ax)
            ax.set_title("Price Trends for the Top 5 Most Expensive Localities")
            ax.set_xlabel("Year Built")
            ax.set_ylabel("Average Property Price (INR)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    elif heading == "Feature Relationship & Correlation":
        st.header("🔗 Feature Relationship & Correlation")
        # Question 11 - Full Width
        st.subheader("11. How are numeric features correlated with each other?")
        numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
        correlation_data = df[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(correlation_data,annot=True,fmt=".2f",cmap="coolwarm",center=0,ax=ax)
        ax.set_title("Correlation Heatmap of Numeric Features")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        # Question 12 and Question 13
        col1, col2 = st.columns(2)
        # Question 12
        with col1:
            st.subheader("12. How do nearby schools relate to price per sq ft?")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df,x="Nearby_Schools",y="Price_per_SqFt",alpha=0.5,ax=ax)
            ax.set_title("Nearby Schools vs Price per Sq Ft")
            ax.set_xlabel("Nearby Schools")
            ax.set_ylabel("Price per Sq Ft (INR)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        # Question 13
        with col2:
            st.subheader("13. How do nearby hospitals relate to price per sq ft?")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df,x="Nearby_Hospitals",y="Price_per_SqFt",alpha=0.5,ax=ax)
            ax.set_title("Nearby Hospitals vs Price per Sq Ft")
            ax.set_xlabel("Nearby Hospitals")
            ax.set_ylabel("Price per Sq Ft (INR)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    # Question 14 and Question 15
        col3, col4 = st.columns(2)
    # Question 14
        with col3:
            st.subheader("14. How does price vary by furnished status?")
            furnished_data = df[["Furnished_Status","Price"]].dropna()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=furnished_data,x="Furnished_Status",y="Price",ax=ax)
            ax.set_title("Property Price by Furnished Status")
            ax.set_xlabel("Furnished Status")
            ax.set_ylabel("Property Price (INR)")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    # Question 15
        with col4:
            st.subheader("15. How does price per sq ft vary by property facing direction?")
            facing_data = df[["Facing","Price_per_SqFt"]].dropna()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=facing_data,x="Facing",y="Price_per_SqFt",ax=ax)
            ax.set_title("Price per Sq Ft by Facing Direction")
            ax.set_xlabel("Facing Direction")
            ax.set_ylabel("Price per Sq Ft (INR)")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    elif heading == "Investment / Amenities / Ownership Analysis":
        st.header("🏠 Investment / Amenities / Ownership Analysis")
        # Question 16 and Question 17
        col1, col2 = st.columns(2)
        # Question 16
        with col1:
            st.subheader("16. How many properties belong to each owner type?")
            owner_data = (df["Owner_Type"].value_counts().reset_index())
            owner_data.columns = ["Owner_Type","Property_Count"]
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=owner_data,x="Owner_Type",y="Property_Count",ax=ax)
            ax.set_title("Properties by Owner Type")
            ax.set_xlabel("Owner Type")
            ax.set_ylabel("Number of Properties")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        # Question 17
        with col2:
            st.subheader("17. How many properties are available under each availability status?")
            availability_data = (df["Availability_Status"].value_counts().reset_index())
            availability_data.columns = ["Availability_Status","Property_Count"]
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=availability_data,x="Availability_Status",y="Property_Count",ax=ax)
            ax.set_title("Properties by Availability Status")
            ax.set_xlabel("Availability Status")
            ax.set_ylabel("Number of Properties")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        # Question 18 and Question 19
        col3, col4 = st.columns(2)
        # Question 18
        with col3:
            st.subheader("18. Does parking space affect property price?")
            parking_data = df[["Parking_Space","Price"]].dropna()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=parking_data,x="Parking_Space",y="Price",ax=ax)
            ax.set_title("Property Price by Parking Availability")
            ax.set_xlabel("Parking Space")
            ax.set_ylabel("Property Price (INR)")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        # Question 19
        with col4:
            st.subheader("19. How do amenities affect price per sq ft?")
            amenity_columns = ["Amenity_Garden","Amenity_Gym","Amenity_Playground","Amenity_Clubhouse","Amenity_Pool"]
            available_amenities = [column for column in amenity_columns if column in df.columns]
            amenity_results = []
            for amenity in available_amenities:
                average_price = df.groupby(amenity)["Price_per_SqFt"].mean().reset_index()
                average_price["Amenity"] = amenity.replace("Amenity_","")
                average_price = average_price.rename(
                columns={amenity: "Available","Price_per_SqFt": "Average_Price_per_SqFt"})
                amenity_results.append(average_price)
            if amenity_results:
                amenity_data = pd.concat(amenity_results,ignore_index=True)
                amenity_data["Available"] = amenity_data["Available"].map({0: "No",1: "Yes"})
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=amenity_data,x="Amenity",y="Average_Price_per_SqFt",hue="Available",ax=ax)
                ax.set_title("Amenities vs Average Price per Sq Ft")
                ax.set_xlabel("Amenity")
                ax.set_ylabel("Average Price per Sq Ft (INR)")
                ax.tick_params(axis="x", rotation=30)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("Amenity columns were not found in the dataset.")
        # Question 20 - Centered
        left_space, middle_col, right_space = st.columns([1, 2, 1])
        with middle_col:
            st.subheader("20. How does public transport accessibility relate to price per sq ft or investment potential?")
            transport_data = df[["Public_Transport_Accessibility","Price_per_SqFt"]].dropna()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=transport_data,x="Public_Transport_Accessibility",y="Price_per_SqFt",ax=ax)
            ax.set_title("Public Transport Accessibility vs Price per Sq Ft")
            ax.set_xlabel("Public Transport Accessibility")
            ax.set_ylabel("Price per Sq Ft (INR)")
            ax.tick_params(axis="x", rotation=30) 
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    