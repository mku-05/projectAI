import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Define the deep learning model for fraud detection
class FraudDetectionModel(nn.Module):
    def __init__(self, input_size):
        super(FraudDetectionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x


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

losses = []


# Function to train the model
def train_model(df):
    # Split dataset into features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    # Create DataLoader for batching
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = FraudDetectionModel(input_size=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    # Training loop
    for epoch in range(10):
        epoch_loss = 0
        # Set number of epochs
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

    model.eval()
    with torch.no_grad():  # No need to compute gradients for evaluation
        predictions = model(X_train_tensor)
        predicted_classes = (predictions > 0.5).float()
    accuracy = accuracy_score(y_train_tensor, predicted_classes)
    precision = precision_score(y_train_tensor, predicted_classes)
    recall = recall_score(y_train_tensor, predicted_classes)
    f1 = f1_score(y_train_tensor, predicted_classes)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    # Plotting the loss curve
    plt.plot(range(1, 11), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()
    return model


# Main execution
if __name__ == "__main__":
    # Train the model on synthetic data
    model = train_model(df)