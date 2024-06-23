from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Muat model
model_filename = 'gradient_boosting_model.pkl'
model = joblib.load(model_filename)

# Load the trained model
model_reg = joblib.load("best_random_forest_model.pkl")

# Definisikan aplikasi FastAPI
app = FastAPI()

# Definisikan struktur input data menggunakan Pydantic
class CustomerData(BaseModel):
    Recency: int
    Frequency: int
    Monetary: float
    R_Score: int
    F_Score: int
    M_Score: int
    RFM_Score: int
    Segment: str

# Define the input data model
class InputData(BaseModel):
    Recency: int
    Frequency: int
    Density: int
    Monetary: float
    R_Score: int
    F_Score: int
    D_Score: int
    M_Score: int
    RFM_Score: int
    AOV: float
    profit_margin: float
    CLV: float
    FutureRevenue: float
    Recency_Frequency: int
    Recency_Monetary: float
    Frequency_Monetary: float
    AvgTransactionValue: float

# Endpoint GET di root
@app.get("/", summary="Read Root", response_description="Check if the API is running.")
async def read_root():
    return {"message": "API for Customer Lifetime Value (CLV) prediction is running."}

# Endpoint untuk prediksi
@app.post("/predict_rfm_segement/")
def predict(data: CustomerData):
    # Konversi data input menjadi format DataFrame yang sesuai untuk model
    input_data = pd.DataFrame([{
        'Recency': data.Recency,
        'Frequency': data.Frequency,
        'Monetary': data.Monetary,
        'R_Score': data.R_Score,
        'F_Score': data.F_Score,
        'M_Score': data.M_Score,
        'RFM_Score': data.RFM_Score,
        'Segment': data.Segment
    }])
    
    # Lakukan prediksi
    prediction = model.predict(input_data)
    
    # Kembalikan hasil prediksi
    return {"RFM_Segment": prediction[0]}

@app.post("/predict_CLV_Future/")
async def predict(data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Perform prediction using the loaded model
    try:
        prediction = model_reg.predict(input_df)
        return {"Future_CLV": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
