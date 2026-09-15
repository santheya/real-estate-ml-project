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