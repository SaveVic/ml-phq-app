# File: train_classifier.py
# Tujuan: Membaca file features.csv dan melatih model klasifikasi ringan.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import argparse
import joblib


class EmotionFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),  # Layer pertama lebih lebar
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # Tambah satu hidden layer
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x): return self.layers(x)


def get_dataloaders(csv_path, batch_size=128, test_split=0.2):
    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=test_split, random_state=42, stratify=y_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, './runs/delta_scaler.pkl')
    print("✅ Scaler berhasil disimpan ke scaler.pkl")

    train_ds = EmotionFeatureDataset(X_train, y_train)
    val_ds = EmotionFeatureDataset(X_val, y_val)
    test_ds = EmotionFeatureDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X_train.shape[1], len(np.unique(y))


def train_and_evaluate(config):
    train_loader, val_loader, test_loader, input_dim, num_classes = get_dataloaders(
        config.csv_path, config.batch_size
    )

    model = MLPClassifier(input_dim, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    best_val_loss = float('inf')  # Inisialisasi dengan nilai tak terhingga
    print(f"🚀 Memulai Training untuk {config.num_epochs} epoch di {device}...")
    for epoch in range(config.num_epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)  # Update scheduler
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Simpan model hanya jika validation loss saat ini lebih baik
            torch.save(model.state_dict(), './runs/best_emotion_classifier.pth')
            print(
                f"✨ Model terbaik baru ditemukan! Val Loss: {avg_val_loss:.4f}. Disimpan ke './runs/best_emotion_classifier.pth'")
        if (epoch + 1) % 10 == 0:
            acc = accuracy_score(val_labels, val_preds)
            f1 = f1_score(val_labels, val_preds, average='macro')
            print(f"Epoch [{epoch+1}/{config.num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Acc: {acc:.4f}, Val F1: {f1:.4f}")

    print("\n🧪 Menjalankan Testing pada model TERBAIK...")
    model.load_state_dict(torch.load('./runs/best_emotion_classifier.pth'))
    model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            # ... (logika testing tetap sama) ...
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    print("\n--- Laporan Klasifikasi Final (dari model terbaik) ---")
    print(classification_report(test_labels, test_preds, digits=4))
    # torch.save(model.state_dict(), 'emotion_classifier.pth')
    return input_dim, num_classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Klasifikasi Emosi dari Fitur Geometris.")
    parser.add_argument("--csv_path", type=str, default="emotion_features.csv",
                        help="Path ke file CSV hasil preprocessing.")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)

    args = parser.parse_args()
    train_and_evaluate(args)
