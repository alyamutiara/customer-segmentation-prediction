# Authenticate and setup Google Cloud SDK
from google.colab import auth

from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# # Set up the BigQuery client
# project_id = 'finalproject-g2df12'
# client = bigquery.Client(project=project_id)

# # Load data from BigQuery
# query = """
# SELECT *
# FROM `finalproject-g2df12.retail.transaction_cleaned`
# """

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

# Calculate quantiles for segmentation
quantiles = rfm['RFM_Score'].quantile([0.33, 0.66]).values

# Define a function to assign segments based on quantiles
def segment_rfm(score):
    if score <= quantiles[0]:
        return 'Low'
    elif score <= quantiles[1]:
        return 'Mid'
    else:
        return 'High'

# Apply the segmentation function
rfm['Segment'] = rfm['RFM_Score'].apply(segment_rfm)