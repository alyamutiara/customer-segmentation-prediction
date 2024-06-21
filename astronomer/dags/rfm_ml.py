from airflow.decorators import dag
from datetime import datetime
from airflow.operators.python import PythonOperator
import pandas as pd
from google.oauth2 import service_account
from google.cloud import storage, bigquery
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pickle
import numpy as np

# Define the DAG
@dag(
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['rfm_ml'],
)
def rfm_ml():
    
    credentials = service_account.Credentials.from_service_account_file(
        '/usr/local/airflow/include/gcp/service_account.json',
    )

    def query_bigquery():
        query = """
        SELECT *
        FROM `iconic-indexer-418610.retail.transaction_cleaned`
        """
        project_id = "iconic-indexer-418610"
        return pd.read_gbq(query, project_id=project_id, dialect="standard", credentials=credentials)

    def perform_rfm_segmentation():
        rfm_df = query_bigquery()  # Call the function to get DataFrame

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
            r'[45]{3}': 'Champions',
            r'[345]{3}': 'Loyal Customers',
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

        # Save to CSV file
        rfm.to_csv('/usr/local/airflow/models/rfm_segments.csv', index=False)

    def upload_to_gcs(file_path, bucket_name, blob_name):
        # Initialize a client for Google Cloud Storage
        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)

    def upload_rfm_segments_to_gcs():
        upload_to_gcs('/usr/local/airflow/models/rfm_segments.csv', 'project_online_retail', 'rfm_segments/rfm_segments.csv')

    def load_to_bigquery():
        # Initialize a client for BigQuery
        client = bigquery.Client(credentials=credentials, project="iconic-indexer-418610")
        dataset_id = 'retail'
        table_id = 'rfm_segments_final'

        # Load the CSV data into BigQuery
        table_ref = client.dataset(dataset_id).table(table_id)
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            autodetect=True,
        )

        with open('/usr/local/airflow/models/rfm_segments.csv', 'rb') as source_file:
            job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

        job.result()  # Wait for the job to complete

    def train_and_save_models():
        # Load the segmented data
        rfm = pd.read_csv('/usr/local/airflow/models/rfm_segments.csv')

        # Feature Engineering
        # Convert segment into numerical values using one-hot encoding
        column_transformer = ColumnTransformer([
            ('segment', OneHotEncoder(), ['Segment'])
        ], remainder='passthrough')

        # Define features and target variable
        X = rfm[['R_Score', 'F_Score', 'M_Score', 'Segment']]
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
            model_performance[model_name] = {'MSE': mse, 'R2': r2, 'CV_MSE': cv_mean}
            
            # Save the trained model
            model_filename = f'/usr/local/airflow/models/{model_name.replace(" ", "_").lower()}_model.pkl'
            with open(model_filename, 'wb') as model_file:
                pickle.dump(pipeline, model_file)

    def upload_model_files_to_gcs():
        models = ['linear_regression_model.pkl', 'random_forest_model.pkl', 'decision_tree_model.pkl', 'support_vector_regressor_model.pkl']
        for model in models:
            upload_to_gcs(f'/usr/local/airflow/models/{model}', 'project_online_retail', f'models/{model}')

    # Define the tasks
    run_query_task = PythonOperator(
        task_id='query_bigquery',
        python_callable=query_bigquery,
    )

    perform_rfm_segmentation_task = PythonOperator(
        task_id='perform_rfm_segmentation',
        python_callable=perform_rfm_segmentation,
    )

    upload_rfm_segments_to_gcs_task = PythonOperator(
        task_id='upload_rfm_segments_to_gcs',
        python_callable=upload_rfm_segments_to_gcs,
    )

    load_to_bigquery_task = PythonOperator(
        task_id='load_to_bigquery',
        python_callable=load_to_bigquery,
    )

    train_and_save_models_task = PythonOperator(
        task_id='train_and_save_models',
        python_callable=train_and_save_models,
    )

    upload_model_files_to_gcs_task = PythonOperator(
        task_id='upload_model_files_to_gcs',
        python_callable=upload_model_files_to_gcs,
    )

    # Set task dependencies
    run_query_task >> perform_rfm_segmentation_task >> upload_rfm_segments_to_gcs_task >> load_to_bigquery_task >> train_and_save_models_task >> upload_model_files_to_gcs_task

rfm_ml()
