import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


###creating the data
df = pd.read_csv("G:/Project/Realestate_project/Scripts/property_data_updated.csv")
X = df[["State","City","Locality","Property_Type","BHK","Size_in_SqFt","Year_Built","Furnished_Status","Floor_No","Total_Floors","Age_of_Property","Nearby_Schools","Nearby_Hospitals","Public_Transport_Accessibility","Parking_Space","Security","Amenities","Facing","Owner_Type","Availability_Status"]]
Y = df["Price_in_Lakhs"]
###split the data
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=42)

###encoding
Numerical_columns = ["BHK","Size_in_SqFt","Year_Built","Floor_No","Total_Floors","Age_of_Property","Nearby_Schools","Nearby_Hospitals"]
Categorical_columns = ["State","City","Locality","Property_Type","Furnished_Status","Public_Transport_Accessibility","Parking_Space","Security","Amenities","Facing","Owner_Type","Availability_Status"]
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
tree_model = DecisionTreeRegressor(random_state=42)
#Train the model
tree_model.fit(x_train_processed, y_train)
##Predict the model
y_pred_DT = tree_model.predict(x_test_processed)
#Evaluate the model
mae_DT = mean_absolute_error(y_test,y_pred_DT)
mse_DT = mean_squared_error(y_test,y_pred_DT)
rmse_DT = mse**0.5
r2_DT = r2_score(y_test,y_pred_DT)
print("Decision Tree MAE:", mae_DT)
print("Decision Tree MSE:", mse_DT)
print("Decision Tree RMSE:", rmse_DT)
print("Decision Tree R2 Score:", r2_DT) 


####3.Random Tree model
###Create Decision Tree Model
rf_model = RandomForestRegressor(n_estimators=50,random_state=42,n_jobs=-1)
#Train the model
rf_model.fit(x_train_processed, y_train)
##Predict the model
y_pred_rf = rf_model.predict(x_test_processed)
#Evaluate the model
mae_rf = mean_absolute_error(y_test,y_pred_rf)
mse_rf = mean_squared_error(y_test,y_pred_rf)
rmse_rf = mse**0.5
r2_rf = r2_score(y_test,y_pred_rf)
print("Random_Forest MAE:", mae_rf)
print("Random_Forest MSE:", mse_rf)
print("Random_Forest RMSE:", rmse_rf)
print("Random_Forest R2 Score:", r2_rf)


