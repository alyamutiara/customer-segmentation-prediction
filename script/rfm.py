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

# Set up the BigQuery client
project_id = 'finalproject-g2df12'
client = bigquery.Client(project=project_id)

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
current_date = pd.to_datetime('2012-01-01', utc=True)

# Calculate the difference between the current date and the invoice date
rfm = rfm_df.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (current_date - date.max()).days,
    'InvoiceNo': lambda num: len(num),
    'TotalPurchase': lambda price: price.sum()
})

# Rename columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']
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

rfm_map = {
    r'[45]{3}': 'Champions', # 555 554 544 545 454 455 445
    r'[345]{3}': 'Loyal Customers', # 543 444 435 355 354 345 344 335
    r'[345][2345][123]': 'Potential Loyalist',
    r'[345][12][12]': 'New Customers',
    r'[345][125][12345]': 'Promising',
    r'[345][234][345]': 'Need Attention',
    r'[23][12345]1': 'About to Sleep',
    r'[12][145][345]': 'Cannot Lose',
    r'[123][2345][2345]': 'At Risk',
    r'[123][123][23]': 'Hibernating',
    r'1[12345][12]': 'Lost Customers'
}

rfm['RFM_Score'] = rfm['RFM_Score'].replace(rfm_map, regex=True)

# Creation of Segment Variable
rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
rfm['RFM_Segment'] = rfm['RFM_Segment'].replace(rfm_map, regex=True)

seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Cant Lose',
    r'3[1-2]': 'About to Sleep',
    r'33': 'Need Attention',
    r'[3-4][4-5]': 'Loyal Customers',
    r'41': 'Promising',
    r'51': 'New Customers',
    r'[4-5][2-3]': 'Potential Loyalists',
    r'5[4-5]': 'Champions'
}

rfm['RFM_Score'] = rfm['RFM_Score'].replace(seg_map, regex=True)

# Creation of Segment Variable
rfm['RF_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str)
rfm['RF_Segment'] = rfm['RF_Segment'].replace(seg_map, regex=True)
rfm.head()