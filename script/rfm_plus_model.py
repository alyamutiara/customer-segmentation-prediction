# Install necessary libraries
pip install google-cloud-bigquery
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn

# Authenticate and setup Google Cloud SDK
from google.colab import auth

from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the BigQuery client
project_id = 'finalproject-g2df12'
client = bigquery.Client(project=project_id)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data from BigQuery
query = """
SELECT *
FROM `finalproject-g2df12.retail.transaction_cleaned`
"""

# Load data into a DataFrame
rfm_df = client.query(query).to_dataframe()

# Perform RFM segmentation
# Calculate total purchase each transaction
rfm_df['TotalPurchase'] = rfm_df['UnitPrice'] * rfm_df['Quantity']

# Define the current date
current_date = rfm_df['InvoiceDate'].max() + pd.DateOffset(1)

# Calculate the difference between the current date and the invoice date
rfm = rfm_df.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (current_date - date.max()).days,
    'InvoiceNo': lambda num: len(num),
    'Quantity': lambda quant: quant.sum(),
    'TotalPurchase': lambda price: price.sum()
})

# Rename columns
rfm.columns = ['Recency', 'Frequency', 'Density', 'Monetary']
rfm = rfm.reset_index()

# Define RFM score calculation
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
rfm['F_Score'] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

# Calculate RFM score
rfm['RFM_Score'] = (rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)).astype(int)

# Import the necessary libraries
import re

# Define the mapping function
def map_segment(rfm_score):
    seg_map = {
        r'111|112|121|131|141|151': 'Lost customers',
        r'332|322|233|232|223|222|132|123|122|212|211': 'Hibernating customers',
        r'155|154|144|214|215|115|114|113': 'Cannot Lose Them',
        r'255|254|245|244|253|252|243|242|235|234|225|224|153|152|145|143|142|135|134|133|125|124': 'At Risk',
        r'331|321|312|221|213|231|241|251': 'About To Sleep',
        r'535|534|443|434|343|334|325|324': 'Need Attention',
        r'525|524|523|522|521|515|514|513|425|424|413|414|415|315|314|313': 'Promising',
        r'512|511|422|421|412|411|311': 'New Customers',
        r'553|551|552|541|542|533|532|531|452|451|442|441|431|453|433|432|423|353|352|351|342|341|333|323': 'Potential Loyalist',
        r'543|444|435|355|354|345|344|335': 'Loyal',
        r'555|554|544|545|454|455|445': 'Champions'
    }
    for pattern, segment in seg_map.items():
        if re.match(pattern, str(rfm_score)):
            return segment
    return 'Unknown'

# Apply the mapping function
rfm['segment'] = rfm['RFM_Score'].apply(map_segment)

# Display the DataFrame
rfm.head()

# Feature Engineering
# Convert segment into numerical values using one-hot encoding
column_transformer = ColumnTransformer([
    ('segment', OneHotEncoder(), ['segment'])
], remainder='passthrough')

# Define features and target variable
X = rfm[['R_Score', 'F_Score', 'M_Score', 'segment']]
y = rfm['Monetary']  # Assuming 'Monetary' as the CLV here

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler(with_mean=False)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Support Vector Regressor': SVR()
}

# Initialize a dictionary to store model performance
model_performance = {}

# Train and evaluate each model
for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', column_transformer),
        ('scaler', scaler),
        ('regressor', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_mean = np.mean(cv_scores)
    
    # Store the performance
    model_performance[model_name] = {'MSE': mse, 'R2': r2, 'CV_MSE': -cv_mean}
    
    # Save the model using pickle
    with open(f'{model_name.replace(" ", "_").lower()}_model.pkl', 'wb') as file:
        pickle.dump(pipeline, file)

# Display model performance
for model_name, metrics in model_performance.items():
    print(f"{model_name} - MSE: {metrics['MSE']}, R2: {metrics['R2']}, CV_MSE: {metrics['CV_MSE']}")

# Loading a model and making predictions on new data
# Example with Random Forest model
loaded_model_filename = 'random_forest_model.pkl'

with open(loaded_model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Assuming new_data is the new dataset for prediction
# new_data = ...

# Predicting CLV for new data
# new_data_predictions = loaded_model.predict(new_data)

# Display predictions
# print(new_data_predictions)