import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from eegnet import EEGNet
from tsgl_loss import TSGLoss

import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    seed = 46  # Change this for each run: 42, 43, 44, 45, 46
    set_seed(seed)

    # --- Load your preprocessed data here ---
    X = np.load('data/X_trainval.npy')
    y = np.load('data/y_trainval.npy')

    # Split into train and validation sets (80/20 split)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Create DataLoaders (set num_workers=0 for Windows compatibility)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # --- Model, Loss, Optimizer ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EEGNet(n_channels=60, n_samples=500, n_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    tsgl_loss = TSGLoss(beta1=1e-4, beta2=1e-4, beta3=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- Training Loop ---
    epochs = 20
    best_val_acc = 0
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            ce = criterion(logits, y_batch)
            loss = tsgl_loss(model, ce)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = running_loss / total
        acc = correct / total

        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch)
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == y_batch).sum().item()
                val_total += y_batch.size(0)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {acc:.4f} | Val Acc: {val_acc:.4f}")

        # --- Early stopping ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"eegnet_tsgl_best_seed{seed}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # --- Save model ---
    torch.save(model.state_dict(), f"eegnet_tsgl_last_seed{seed}.pth")