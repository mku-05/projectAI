# train_model.py
import pandas as pd

from data_preprocessing import preprocess_data
from model import train_model
import torch

if __name__ == "__main__":
    # Generate synthetic data for fraud detection
    data = [
        [2500.00, 1, 14, 102, 1002, 1, 10],
        [15.00, 0, 9, 103, 1003, 0, 2],
        [5000.00, 1, 17, 104, 1004, 2, 1],
        [100.00, 0, 11, 105, 1005, 0, 12],
        [300.00, 1, 21, 106, 1006, 1, 3],
        [1200.00, 0, 8, 107, 1007, 2, 15],
        [200.00, 0, 19, 108, 1008, 0, 8],
        [150.00, 1, 5, 109, 1009, 1, 20],
        [2000.00, 0, 13, 110, 1010, 0, 6],
        [300.00, 1, 10, 111, 1011, 2, 11],
        [75.00, 0, 16, 112, 1012, 1, 25],
        [500.00, 0, 22, 113, 1013, 0, 7],
        [125.00, 1, 12, 114, 1014, 1, 9],
        [900.00, 0, 18, 115, 1015, 2, 14],
        [350.00, 1, 23, 116, 1016, 1, 30]
    ]

    # Define feature columns and create the DataFrame
    columns = ['amount', 'transaction_type', 'time_of_day', 'location_code', 'user_id', 'device_type',
               'previous_transactions']
    df = pd.DataFrame(data, columns=columns)

    # Generate synthetic fraud labels (0 = non-fraud, 1 = fraud)
    df['Class'] = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]

    # Train the model
    model = train_model(df)

    # Save the model weights
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved successfully!")
