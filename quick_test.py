#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/jugomez/Library/CloudStorage/Box-Box/Biological_and_Medical_Informatics/Biocomputing_Algorithms_(BMI_203)/Assignments/final-nn')

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from nn.nn import NeuralNetwork

# Load data
print("Loading data...")
digits = load_digits()
X = digits.data
X = X / X.max()

X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")

# Create autoencoder
print("\nCreating autoencoder...")
nn_arch = [
    {'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
    {'input_dim': 16, 'output_dim': 64, 'activation': 'sigmoid'}
]

autoencoder = NeuralNetwork(
    nn_arch=nn_arch,
    lr=0.01,
    seed=42,
    batch_size=32,
    epochs=3,
    loss_function='mean_squared_error'
)

# Test training
print("Starting training...")
y_train = X_train
y_val = X_val
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")

try:
    train_losses, val_losses = autoencoder.fit(X_train, y_train, X_val, y_val)
    print(f"\nTraining successful!")
    print(f"Train losses: {train_losses}")
    print(f"Val losses: {val_losses}")
except Exception as e:
    print(f"\nError during training: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
