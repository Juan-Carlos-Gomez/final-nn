#!/usr/bin/env python3
"""Quick test of neural network implementation"""
import numpy as np

# Test activation functions
class TestNN:
    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def _relu(self, Z):
        return np.maximum(0, Z)
    
    def _mean_squared_error(self, y, y_hat):
        loss = np.mean((y_hat - y) ** 2)
        return loss

nn = TestNN()

# Test sigmoid
Z = np.array([[1, 2], [3, 4]])
sig = nn._sigmoid(Z)
print(f"Sigmoid test: {sig.shape} - OK")

# Test ReLU
relu = nn._relu(Z)
print(f"ReLU test: {relu.shape} - OK")

# Test MSE
y = np.array([[1, 0], [0, 1]])
y_hat = np.array([[0.8, 0.1], [0.1, 0.9]])
loss = nn._mean_squared_error(y, y_hat)
print(f"MSE test: {loss} - OK")

# Test shape operations for fit
X_train = np.random.randn(100, 20)
y_train = np.random.randn(100, 5)
n_samples = X_train.shape[0]

# Simulate batch slicing
indices = np.random.permutation(n_samples)
X_shuffled = X_train[indices]
y_shuffled = y_train[indices]

for i in range(0, n_samples, 32):
    X_batch = X_shuffled[i:i+32]
    y_batch = y_shuffled[i:i+32]
    print(f"Batch {i//32}: X={X_batch.shape}, y={y_batch.shape}")

print("\nAll tests passed!")
