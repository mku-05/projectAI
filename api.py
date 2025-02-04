# api.py

from fastapi import FastAPI
import torch
from pydantic import BaseModel
from model import FraudDetectionModel
import torch

app = FastAPI()


class TransactionData(BaseModel):
    features: list  # This will be a list of feature values for the prediction


# Load the trained model
model = FraudDetectionModel(input_size=7)  # Change input size as per your dataset
model.load_state_dict(torch.load('model.pth'))  # Load the saved model weights
model.eval()


@app.post("/predict")
def predict(data: TransactionData):
    input_data = torch.tensor([data.features], dtype=torch.float32)
    prediction = model(input_data).item()
    return {"fraud_probability": prediction}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
