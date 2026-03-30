"""
ann_model.py
============
ANN (Artificial Neural Network) for MNIST Classification

Architecture:
    Input   (N, 784)
    FC1     784 -> 256, ReLU
    FC2     256 -> 128, ReLU
    FC3     128 -> 10,  Softmax

No external libraries — numpy only.
"""

import numpy as np


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def relu(x):
    return np.maximum(0, x)


def relu_backward(dout, x):
    return dout * (x > 0)


def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x   = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def cross_entropy_loss(y_pred, y_true):
    """
    Cross-entropy loss and its gradient w.r.t. the softmax input.

    Parameters
    ----------
    y_pred : np.ndarray, shape (N, 10) — softmax probabilities
    y_true : np.ndarray, shape (N,)    — integer class labels 0-9

    Returns
    -------
    loss : float
    grad : np.ndarray, shape (N, 10)
    """
    N              = y_pred.shape[0]
    y_pred_clipped = np.clip(y_pred, 1e-12, 1.0)
    loss           = -np.sum(np.log(y_pred_clipped[np.arange(N), y_true])) / N
    grad           = y_pred_clipped.copy()
    grad[np.arange(N), y_true] -= 1
    grad          /= N
    return loss, grad


# =============================================================================
# FULLY CONNECTED LAYER
# =============================================================================

class FCLayer:
    """
    Fully-connected layer: out = x @ W + b

    Weights are He-initialised. Momentum velocity terms are tracked
    internally and updated each call to update().
    """

    def __init__(self, in_features, out_features):
        self.in_features  = in_features
        self.out_features = out_features

        self.W  = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b  = np.zeros(out_features)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

        self.x  = None   # cached for backward pass
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T

    def update(self, lr, momentum=0.9):
        self.vW = momentum * self.vW - lr * self.dW
        self.vb = momentum * self.vb - lr * self.db
        self.W += self.vW
        self.b += self.vb


# =============================================================================
# ANN MODEL
# =============================================================================

class ANN:
    """
    784 -> FC1(256) -> ReLU -> FC2(128) -> ReLU -> FC3(10) -> Softmax
    """

    def __init__(self):
        self.fc1 = FCLayer(784, 256)
        self.fc2 = FCLayer(256, 128)
        self.fc3 = FCLayer(128, 10)

        self.relu1_input = None
        self.relu2_input = None

    def forward(self, x):
        """
        Parameters
        ----------
        x : np.ndarray, shape (N, 784)

        Returns
        -------
        probs : np.ndarray, shape (N, 10)
        """
        x = self.fc1.forward(x)
        self.relu1_input = x
        x = relu(x)

        x = self.fc2.forward(x)
        self.relu2_input = x
        x = relu(x)

        x = self.fc3.forward(x)
        return softmax(x)

    def backward(self, grad):
        """
        Parameters
        ----------
        grad : np.ndarray, shape (N, 10) — gradient from cross_entropy_loss
        """
        grad = self.fc3.backward(grad)
        grad = relu_backward(grad, self.relu2_input)
        grad = self.fc2.backward(grad)
        grad = relu_backward(grad, self.relu1_input)
        grad = self.fc1.backward(grad)

    def update(self, lr, momentum=0.9):
        self.fc1.update(lr, momentum)
        self.fc2.update(lr, momentum)
        self.fc3.update(lr, momentum)

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)
