import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report  # For evaluation and plotting

class ActivityDataset(Dataset):
    """Custom dataset class"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class Attention(nn.Module):
    """Simple Self-Attention Mechanism"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(hidden_size, 1))

    def forward(self, lstm_out):
        # lstm_out shape: (batch_size, seq_len, hidden_size * 2)
        # Calculate attention scores (batch_size, seq_len, 1)
        scores = torch.matmul(lstm_out, self.attention_weights)  # (batch_size, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch_size, seq_len)

        # Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)  # (batch_size, seq_len)
        
        # Weighted sum of LSTM outputs
        weighted_sum = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_size * 2)
        
        return weighted_sum, attention_weights

class CNNLSTMWithAttention(nn.Module):
    def __init__(self, input_size=3, cnn_out_channels=256, kernel_size=3, lstm_hidden_size=256, num_lstm_layers=3, num_classes=10, dropout=0.2):
        super(CNNLSTMWithAttention, self).__init__()

        # CNN part
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=cnn_out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(cnn_out_channels)
        self.conv2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels * 2, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(cnn_out_channels * 2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM part
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels * 2,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Attention part
        self.attention = Attention(lstm_hidden_size * 2)  # Bidirectional LSTM, so hidden_size * 2

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # CNN part
        x = x.transpose(1, 2)  # (batch_size, channels, seq_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, channels)

        # LSTM part
        lstm_out, _ = self.lstm(x)

        # Attention part
        attention_out, attention_weights = self.attention(lstm_out)

        # Fully connected layer
        out = self.fc(attention_out)
        return out, attention_weights  # Return predictions and attention weights


def load_and_preprocess_data(base_dir, window_size=128):
    """Load and preprocess data"""
    sequences = []
    labels = []
    label_counts = {}  # Count sequences for each label

    # Iterate over folders for each activity type
    for activity_id in range(2806, 2816):  # 2806-2815
        activity_dir = os.path.join(base_dir, str(activity_id))
        if not os.path.exists(activity_dir):
            continue
            
        print(f"Processing activity {activity_id}...")
        
        # Read all CSV files in the folder
        for file in os.listdir(activity_dir):
            if not file.endswith('.csv'):
                continue
                
            file_path = os.path.join(activity_dir, file)
            try:
                # Read CSV file
                data = pd.read_csv(
                    file_path,
                    header=None,
                    names=['RandomID', 'Timestamp', 'Accel_X', 'Accel_Y', 'Accel_Z']
                )
                
                # Extract acceleration data
                sensor_data = data[['Accel_X', 'Accel_Y', 'Accel_Z']].values
                
                # Use sliding window to segment data
                for i in range(0, len(sensor_data) - window_size + 1, window_size // 2):
                    sequence = sensor_data[i:i + window_size]
                    if len(sequence) == window_size:
                        sequences.append(sequence)
                        labels.append(activity_id - 2806)  # Map activity ID to 0-9
                        label = activity_id - 2806

                        # Update label counts
                        if label not in label_counts:
                            label_counts[label] = 0
                        label_counts[label] += 1
                        
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue

    # Print sequence counts for each label
    print("Label counts (number of sequences for each label):")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} sequences")

    return np.array(sequences), np.array(labels)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    """Train and evaluate the model, recording loss history for each epoch"""
    model = model.to(device)
    best_val_acc = 0
    best_model = None  # Save best model weights

    # Loss history lists
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs, _ = model(sequences)  # Get predictions, ignore attention_weights
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_loss_history.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)

                outputs, _ = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_loss_history.append(avg_val_loss)

        # Display training info
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Save best model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()  # Save best model weights
            torch.save(best_model, 'best_lstm_model.pth')
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')

        print('-' * 60)

    # Load best model for evaluation
    print("Loading the best model for evaluation...")
    model.load_state_dict(torch.load('best_lstm_model.pth'))

    print("Evaluating the best model's performance on validation data...")
    model.eval()

    all_labels_best_model = []
    all_predictions_best_model = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs, _ = model(sequences)
            _, predicted = outputs.max(1)

            all_labels_best_model.extend(labels.cpu().numpy())
            all_predictions_best_model.extend(predicted.cpu().numpy())

    # Output classification report
    print("\nBest model performance on each class:")
    print(classification_report(all_labels_best_model, all_predictions_best_model))

    val_acc_final = (np.array(all_predictions_best_model) == np.array(all_labels_best_model)).mean() * 100
    print(f'\nBest model validation accuracy after loading: {val_acc_final:.2f}%')

    # Return loss histories and final predictions for plotting
    return {
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'all_labels': all_labels_best_model,
        'all_predictions': all_predictions_best_model
    }


def main():
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    sequences, labels = load_and_preprocess_data('activity_segments', window_size=128)
    
    # Standardize data
    scaler = StandardScaler()
    sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
    sequences_scaled = scaler.fit_transform(sequences_reshaped)
    sequences = sequences_scaled.reshape(sequences.shape)
    
    # Save scaler
    joblib.dump(scaler, 'lstm_scaler.pkl')
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create data loaders
    train_dataset = ActivityDataset(X_train, y_train)
    test_dataset = ActivityDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = CNNLSTMWithAttention(
        input_size=3,          # Three-axis acceleration data
        lstm_hidden_size=128,  # LSTM hidden size
        num_lstm_layers=2,     # Number of LSTM layers
        num_classes=10         # 10 activity types
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Starting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=450,
        device=device
    )
    
    print("Training completed!")

    # Plot and save Loss Curve
    epochs = range(1, len(history['train_loss_history']) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_loss_history'], label='Training Loss', marker='o')
    plt.plot(epochs, history['val_loss_history'], label='Validation Loss', marker='o')
    plt.title('Loss Curve - DeepConvLSTM-Attention Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    plt.show()

    # Plot and save Confusion Matrix
    cm = confusion_matrix(history['all_labels'], history['all_predictions'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - DeepConvLSTM-Attention Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Plot and save Classification Report
    report_dict = classification_report(history['all_labels'], history['all_predictions'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    if 'accuracy' in report_df.index:
        report_df = report_df.drop('accuracy')
    plt.figure(figsize=(10, 6))
    if 'support' in report_df.columns:
        sns.heatmap(report_df.drop(columns=['support']), annot=True, cmap='YlGnBu')
    else:
        sns.heatmap(report_df, annot=True, cmap='YlGnBu')
    plt.title('Classification Report - DeepConvLSTM-Attention')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.tight_layout()
    plt.savefig('classification_report.png')
    plt.show()


if __name__ == "__main__":
    main() 
