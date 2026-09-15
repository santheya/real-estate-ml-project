import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib

###creating the data
df = pd.read_csv("G:/Project/Realestate_project/Scripts/property_data_updated.csv")
X = df.drop(columns=["ID","Price","Price_in_Lakhs","Amenities","Future_Price_5Y","Good_Investment"])
Y = df["Future_Price_5Y"]

###split the data
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=42)

###encoding
Numerical_columns = ["BHK","Size_in_SqFt","Year_Built","Floor_No","Total_Floors","Age_of_Property","Nearby_Schools","Nearby_Hospitals","Amenity_Garden","Amenity_Gym","Amenity_Playground","Amenity_Clubhouse","Amenity_Pool"]
Categorical_columns = ["State","City","Locality","Property_Type","Furnished_Status","Public_Transport_Accessibility","Parking_Space","Security","Facing","Owner_Type","Availability_Status"]
preprocessor = ColumnTransformer ( transformers= [("num",StandardScaler(),Numerical_columns),("cat",OneHotEncoder(handle_unknown="ignore"),Categorical_columns)])

#Train the model
preprocessor.fit(x_train)
x_train_processed = preprocessor.transform(x_train)
x_test_processed = preprocessor.transform(x_test)

#####1. Linear Regression Model
#Create the LinerRegression model
model=LinearRegression()
#Train the model
model.fit(x_train_processed, y_train)
##Predict the model
y_pred = model.predict(x_test_processed)
#Evaluate the model
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = mse**0.5
r2 = r2_score(y_test,y_pred)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

#### 2.Decision Tree Model
###Create Decision Tree Model
tree_model = DecisionTreeRegressor(random_state=42,max_depth=10)
#Train the model
tree_model.fit(x_train_processed, y_train)
##Predict the model
y_pred_DT = tree_model.predict(x_test_processed)
#Evaluate the model
mae_DT = mean_absolute_error(y_test,y_pred_DT)
mse_DT = mean_squared_error(y_test,y_pred_DT)
rmse_DT = mse_DT**0.5
r2_DT = r2_score(y_test,y_pred_DT)
print("Decision Tree MAE:", mae_DT)
print("Decision Tree MSE:", mse_DT)
print("Decision Tree RMSE:", rmse_DT)
print("Decision Tree R2 Score:", r2_DT) 

####3.Random Forest model
###Create Random Forest Model
rf_model = RandomForestRegressor(n_estimators=50,max_depth=10,random_state=42,n_jobs=-1)
#Train the model
rf_model.fit(x_train_processed, y_train)
##Predict the model
y_pred_rf = rf_model.predict(x_test_processed)
#Evaluate the model
mae_rf = mean_absolute_error(y_test,y_pred_rf)
mse_rf = mean_squared_error(y_test,y_pred_rf)
rmse_rf = mse_rf**0.5
r2_rf = r2_score(y_test,y_pred_rf)
print("Random_Forest MAE:", mae_rf)
print("Random_Forest MSE:", mse_rf)
print("Random_Forest RMSE:", rmse_rf)
print("Random_Forest R2 Score:", r2_rf)


####4.Gradient boost model
###Creating Model
gb_model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42)
#Train the model
gb_model.fit(x_train_processed, y_train)
##Predict the model
y_pred_gb = gb_model.predict(x_test_processed)
#Evaluate the model
mae_gb = mean_absolute_error(y_test,y_pred_gb)
mse_gb = mean_squared_error(y_test,y_pred_gb)
rmse_gb = mse_gb**0.5
r2_gb = r2_score(y_test,y_pred_gb)
print("Gradient boosting MAE:", mae_gb)
print("Gradient Boosting MSE:", mse_gb)
print("Gradient Boosting RMSE:", rmse_gb)
print("Gradient Boosting R2 Score:", r2_gb)

####5.XG Boost model
###Creating Model
#### 5. XGBoost Model
xgb_model = XGBRegressor(n_estimators=100,learning_rate=0.1,max_depth=6,random_state=42,n_jobs=-1)
#Train the model
xgb_model.fit(x_train_processed, y_train)
##Predict the model
y_pred_xgb = xgb_model.predict(x_test_processed)
#Evaluate the model
mae_xgb = mean_absolute_error(y_test,y_pred_xgb)
mse_xgb = mean_squared_error(y_test,y_pred_xgb)
rmse_xgb = mse_xgb**0.5
r2_xgb = r2_score(y_test,y_pred_xgb)
print("XG Boost MAE:", mae_xgb)
print("XG Boost MSE:", mse_xgb)
print("XG Boost RMSE:", rmse_xgb)
print("XG Boost R2 Score:", r2_xgb)

# Model comparison table

results = {"Model": ["Linear Regression","Decision Tree Regressor","Random Forest Regressor","Gradient Boosting Regressor","XGBoost Regressor"],
    "MAE": [mae,mae_DT,mae_rf,mae_gb,mae_xgb],
    "RMSE": [rmse,rmse_DT,rmse_rf,rmse_gb,rmse_xgb],
    "R2 Score": [r2,r2_DT,r2_rf,r2_gb,r2_xgb]}
results_df = pd.DataFrame(results)
print("\nRegression Model Comparison:")
print(results_df.to_string(index=False))
results_df.to_csv("G:/Project/Realestate_project/Scripts/regression_model_results.csv",index=False)
print("\nRegression results saved successfully.")

# Save the best regression model and preprocessing object
joblib.dump(gb_model, "G:/Project/Realestate_project/Scripts/best_regression_model.pkl")
joblib.dump(preprocessor, "G:/Project/Realestate_project/Scripts/regression_preprocessor.pkl")
print("\nBest regression model saved successfully.")
print("Regression preprocessor saved successfully.")