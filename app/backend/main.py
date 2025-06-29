from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import os
import pandas as pd

class SodAIData(BaseModel):
    customer_type: str
    brand: str
    category: str
    sub_category: str
    segment: str
    package: str
    purchased_last_week: bool
    purchased_last_month: bool
    purchased_ever: bool
    num_deliver_per_week: int
    avg_purchase_period: int
    X: float
    Y: float
    size: float
    weekly_avg_distinct: float


model = load("model.joblib")
app = FastAPI()

data_dir = os.getenv('DATA_DIR', './data')
csv_file_path = os.path.join(data_dir, "predictions.csv")
# en caso de que no exista el directorio, lo creamos
if not os.path.exists(csv_file_path):
    df = pd.DataFrame(columns=[
        "customer_type",
        "brand",
        "category",
        "sub_category",
        "segment",
        "package",
        "purchased_last_week",
        "purchased_last_month",
        "purchased_ever",
        "num_deliver_per_week",
        "avg_purchase_period",
        "X",
        "Y",
        "size",
        "weekly_avg_distinct",
    ])
    df.to_csv(csv_file_path, index=False)

# home
@app.get("/")
def read_root():
    return {"message": "SodAI Predicter API is running!"}

# predict
@app.post("/predict")
def predict(data: SodAIData):
    features_df = pd.DataFrame({
        "customer_type": data.customer_type,
        "brand": data.brand,
        "category": data.category,
        "sub_category": data.sub_category,
        "segment": data.segment,
        "package": data.package,
        "purchased_last_week": int(data.purchased_last_week),
        "purchased_last_month": int(data.purchased_last_month),
        "purchased_ever": int(data.purchased_ever),
        "num_deliver_per_week": data.num_deliver_per_week,
        "avg_purchase_period": data.avg_purchase_period,
        "X": data.X,
        "Y": data.Y,
        "size": data.size,
        "weekly_avg_distinct": data.weekly_avg_distinct,
        })
    label = model.predict(features_df)[0]
    
    new_data = features_df.assign(label=label)
    new_data.to_csv(csv_file_path, mode='a', header=False, index=False)
    
    return {"label": int(label)}
