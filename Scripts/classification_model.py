import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ( accuracy_score,precision_score,recall_score,f1_score,confusion_matrix)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib

###creating the data
df = pd.read_csv("G:/Project/Realestate_project/Scripts/property_data_updated.csv")
X = df.drop(columns=["ID","Price_in_Lakhs","Amenities","Future_Price_5Y","Good_Investment"])
Y = df["Good_Investment"]

###split the data
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=42,stratify=Y)

###encoding
Numerical_columns = ["BHK","Size_in_SqFt","Price_per_SqFt","Year_Built","Floor_No","Total_Floors","Age_of_Property","Nearby_Schools","Nearby_Hospitals","Amenity_Garden","Amenity_Gym","Amenity_Playground","Amenity_Clubhouse","Amenity_Pool","Price"]
Categorical_columns = ["State","City","Locality","Property_Type","Furnished_Status","Public_Transport_Accessibility","Parking_Space","Security","Facing","Owner_Type","Availability_Status"]
preprocessor = ColumnTransformer ( transformers= [("num",StandardScaler(),Numerical_columns),("cat",OneHotEncoder(handle_unknown="ignore"),Categorical_columns)])

#Train the model
preprocessor.fit(x_train)
x_train_processed = preprocessor.transform(x_train)
x_test_processed = preprocessor.transform(x_test)

#####1. Logistic Regression Model
#Create the LinerRegression model
logistic_model = LogisticRegression(max_iter=1000,random_state=42)
#Train the model
logistic_model.fit(x_train_processed, y_train)
##Predict the model
y_pred_logistic = logistic_model.predict(x_test_processed)
#Evaluate the model
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
precision_logistic = precision_score(y_test,y_pred_logistic,zero_division=0)
recall_logistic = recall_score(y_test,y_pred_logistic,zero_division=0)
f1_logistic = f1_score(y_test,y_pred_logistic,zero_division=0)
cm_logistic = confusion_matrix(y_test, y_pred_logistic)
print("Logistic Regression Accuracy:", accuracy_logistic)
print("Logistic Regression Precision:", precision_logistic)
print("Logistic Regression Recall:", recall_logistic)
print("Logistic Regression F1 Score:", f1_logistic)
print("Confusion Matrix:")
print(cm_logistic)

##### 2. Decision Tree Classification Model
# Create the Decision Tree Classifier model
DTC_model = DecisionTreeClassifier(max_depth=10,random_state=42)
#Train the model
DTC_model.fit(x_train_processed, y_train)
##Predict the model
y_pred_DTC = DTC_model.predict(x_test_processed)
#Evaluate the model
accuracy_DTC = accuracy_score(y_test, y_pred_DTC)
precision_DTC = precision_score(y_test,y_pred_DTC,zero_division=0)
recall_DTC = recall_score(y_test,y_pred_DTC,zero_division=0)
f1_DTC = f1_score(y_test,y_pred_DTC,zero_division=0)
cm_DTC = confusion_matrix(y_test, y_pred_DTC)
print("Decision Tree Accuracy:", accuracy_DTC)
print("Decision Tree Precision:", precision_DTC)
print("Decision Tree Recall:", recall_DTC)
print("Decision Tree F1 Score:", f1_DTC)
print("Confusion Matrix:")
print(cm_DTC)

#####3.Random forest classification Model
#Create the Random Forest classifier model
RF_model = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,n_jobs=-1)
#Train the model
RF_model.fit(x_train_processed, y_train)
##Predict the model
y_pred_rf = RF_model.predict(x_test_processed)
#Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test,y_pred_rf,zero_division=0)
recall_rf = recall_score(y_test,y_pred_rf,zero_division=0)
f1_rf = f1_score(y_test,y_pred_rf,zero_division=0)
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print("Random Forest Precision:", precision_rf)
print("Random Forest Recall:", recall_rf)
print("Random Forest F1 Score:", f1_rf)
print("Confusion Matrix:")
print(cm_rf)

#####4.Gradient Boosting Classifier Model
#Create theGradient Boosting Classifier model
GB_model = GradientBoostingClassifier(n_estimators=100,max_depth=10,random_state=42)
#Train the model
GB_model.fit(x_train_processed, y_train)
##Predict the model
y_pred_gb = GB_model.predict(x_test_processed)
#Evaluate the model
accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test,y_pred_gb,zero_division=0)
recall_gb = recall_score(y_test,y_pred_gb,zero_division=0)
f1_gb = f1_score(y_test,y_pred_gb,zero_division=0)
cm_gb = confusion_matrix(y_test, y_pred_gb)
print("Gradient Boosting Accuracy:", accuracy_gb)
print("Gradient Boosting Precision:", precision_gb)
print("Gradient Boosting Recall:", recall_gb)
print("Gradient Boosting F1 Score:", f1_gb)
print("Confusion Matrix:")
print(cm_gb)

#####5.XGBoost Classification Model
#Create the XGBoost Classifier. model
XG_model = XGBClassifier(n_estimators=100,max_depth=6,learning_rate=0.1,random_state=42,n_jobs=-1)
#Train the model
XG_model.fit(x_train_processed, y_train)
##Predict the model
y_pred_xg = XG_model.predict(x_test_processed)
#Evaluate the model
accuracy_xg = accuracy_score(y_test, y_pred_xg)
precision_xg = precision_score(y_test,y_pred_xg,zero_division=0)
recall_xg = recall_score(y_test,y_pred_xg,zero_division=0)
f1_xg = f1_score(y_test,y_pred_xg,zero_division=0)
cm_xg = confusion_matrix(y_test, y_pred_xg)
print("XG Boosting Accuracy:", accuracy_xg)
print("XG Boosting Precision:", precision_xg)
print("XG Boosting Recall:", recall_xg)
print("XG Boosting F1 Score:", f1_xg)
print("Confusion Matrix:")
print(cm_xg)

# --------------------------------------------------
# Classification model comparison
# --------------------------------------------------
classification_results = {
    "Model": ["Logistic Regression","Decision Tree Classifier","Random Forest Classifier","Gradient Boosting Classifier","XGBoost Classifier"],
    "Accuracy": [accuracy_logistic,accuracy_DTC,accuracy_rf,accuracy_gb,accuracy_xg],
    "Precision": [precision_logistic,precision_DTC,precision_rf,precision_gb,precision_xg],
    "Recall": [recall_logistic,recall_DTC,recall_rf,recall_gb,recall_xg],
    "F1 Score": [f1_logistic,f1_DTC,f1_rf,f1_gb,f1_xg]}

classification_results_df = pd.DataFrame(classification_results)
print("\nClassification Model Comparison:")
print(classification_results_df.to_string(index=False))

# --------------------------------------------------
# Save classification results
# --------------------------------------------------
classification_results_df.to_csv("G:/Project/Realestate_project/Scripts/classification_model_results.csv",index=False)
print("\nClassification results saved successfully.")

# --------------------------------------------------
# Save the best classification model
# --------------------------------------------------
joblib.dump(XG_model,"G:/Project/Realestate_project/Scripts/best_classification_model.pkl")
joblib.dump(preprocessor,"G:/Project/Realestate_project/Scripts/classification_preprocessor.pkl")
print("Best classification model saved successfully.")
print("Classification preprocessor saved successfully.")
