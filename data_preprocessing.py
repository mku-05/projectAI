
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Normalize 'Amount' feature
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    # Drop non-relevant columns
    df = df.drop(columns=['Time'])

    return df


if __name__ == "__main__":
    # Example usage
    file_path = "creditcard.csv"
    df = preprocess_data(file_path)
    print(df.head())
