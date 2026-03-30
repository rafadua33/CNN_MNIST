"""
cnn_model.py
============
CNN Model Shell for MNIST Classification
-----------------------------------------
YOUR TASK: Implement every section marked with  *** TODO ***

Everything else has been written for you. Read the docstrings and comments
carefully — they explain exactly what each piece should do.

Architecture you are building:
    Input        (N, 1, 28, 28)   ← N grayscale 28×28 images, channel-first
    ─────────────────────────────────────────────────────
    Conv1        8 filters, 3×3   → (N,  8, 26, 26)
    ReLU
    MaxPool1     2×2, stride 2    → (N,  8, 13, 13)
    ─────────────────────────────────────────────────────
    Conv2        16 filters, 3×3  → (N, 16, 11, 11)
    ReLU
    MaxPool2     2×2, stride 2    → (N, 16,  5,  5)
    ─────────────────────────────────────────────────────
    Flatten                        → (N, 400)
    FC1          400 → 128, ReLU
    FC2          128 → 10,  Softmax  ← one score per digit class
    ─────────────────────────────────────────────────────

Useful formula — output spatial size after a conv or pool:
    out_size = floor((in_size - kernel_size) / stride) + 1   (when padding = 0)
"""

import numpy as np


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def relu(x):
    """
    ReLU (Rectified Linear Unit): replace every negative value with 0.

    Think of it as a gate — positive signals pass through unchanged,
    negative signals are blocked.

        f(x) = max(0, x)   applied element-wise

    Parameters
    ----------
    x : np.ndarray — any shape

    Returns
    -------
    np.ndarray — same shape as x, all negatives zeroed out

    Example
    -------
    relu(np.array([-2, 0, 3])) → [0, 0, 3]
    """
    return np.maximum(0, x)


def relu_backward(dout, x):
    """
    Backprop through ReLU.

    During the forward pass, ReLU zeroed out values where x <= 0.
    During backprop we do the same thing to the gradient:
        - where the original input x was > 0, pass the gradient through unchanged
        - where the original input x was <= 0, block the gradient (set it to 0)

    Parameters
    ----------
    dout : np.ndarray — gradient flowing back from the next layer (same shape as x)
    x    : np.ndarray — the original input that was passed INTO relu() during forward

    Returns
    -------
    np.ndarray — gradient to pass to the layer before this ReLU
    """
    # *** TODO ***
    # Hint: multiply dout by a mask that is 1 where x > 0 and 0 elsewhere
    mask = (x>0)
    return dout * mask


# --- Pre-implemented: softmax is tricky numerically, so it is provided ---

def softmax(x):
    """
    Convert raw scores (logits) into probabilities that sum to 1.

    Uses the numerically stable "subtract the row max" trick to prevent
    overflow when exponentiating large numbers.

    Parameters
    ----------
    x : np.ndarray, shape (N, C)  — N samples, C classes

    Returns
    -------
    np.ndarray, shape (N, C) — each row is a probability distribution
    """
    shifted = x - np.max(x, axis=1, keepdims=True)   # stability trick
    exp_x   = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# =============================================================================
# LOSS FUNCTION  (pre-implemented)
# =============================================================================

def cross_entropy_loss(y_pred, y_true):
    """
    Measures how wrong the model's predictions are.

    Uses cross-entropy:  L = -1/N * sum( log(probability of correct class) )

    Also returns the gradient, which tells the network which direction to
    adjust its weights to reduce the loss.

    The gradient formula for softmax + cross-entropy combined simplifies to:
        grad = (predicted_probabilities - one_hot_true_labels) / N

    Parameters
    ----------
    y_pred : np.ndarray, shape (N, 10) — softmax probabilities from forward pass
    y_true : np.ndarray, shape (N,)    — correct class indices (integers 0–9)

    Returns
    -------
    loss : float          — scalar loss value (lower is better)
    grad : np.ndarray, shape (N, 10) — gradient of loss w.r.t. the softmax input
    """
    N = y_pred.shape[0]

    # Clip to prevent log(0) = -infinity
    y_pred_clipped = np.clip(y_pred, 1e-12, 1.0)

    # Loss: pick out the log-probability of the correct class for each sample
    correct_log_probs = -np.log(y_pred_clipped[np.arange(N), y_true])
    loss = np.sum(correct_log_probs) / N

    # Gradient: one-hot encode then subtract
    grad = y_pred_clipped.copy()
    grad[np.arange(N), y_true] -= 1
    grad /= N

    return loss, grad


# =============================================================================
# CONVOLUTIONAL LAYER
# =============================================================================

class ConvLayer:
    """
    A single 2-D convolutional layer.

    What it does conceptually:
        Slides a small filter (kernel) across the image. At each position the
        filter performs a dot product with the image patch beneath it, producing
        one output value. Doing this across the whole image produces one
        "feature map". Using multiple filters produces multiple feature maps,
        each detecting a different pattern (edges, curves, textures…).

    Weight tensor shape: (num_filters, in_channels, kernel_size, kernel_size)
    """

    def __init__(self, in_channels, num_filters, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding

        # He initialisation — keeps gradient magnitudes healthy at the start
        fan_in  = in_channels * kernel_size * kernel_size
        self.W  = np.random.randn(num_filters, in_channels, kernel_size, kernel_size) \
                  * np.sqrt(2.0 / fan_in)
        self.b  = np.zeros(num_filters)

        # Momentum velocity terms (start at zero, updated during training)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

        # Cached values — set during forward, used during backward
        self.x_padded = None
        self.x_shape  = None

    def forward(self, x):
        """
        Slide every filter across the input and collect the results.

        Parameters
        ----------
        x : np.ndarray, shape (N, C, H, W)

        Returns
        -------
        out : np.ndarray, shape (N, num_filters, H_out, W_out)

        Steps to implement
        ------------------
        1. Pad x with zeros along H and W if self.padding > 0
               np.pad(x, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
        2. Compute output dimensions:
               H_out = (H + 2*padding - kernel_size) // stride + 1
               W_out = (W + 2*padding - kernel_size) // stride + 1
        3. Allocate output array: np.zeros((N, num_filters, H_out, W_out))
        4. Loop:  for each filter f, row i, column j:
               patch = x_padded[:, :, i*stride : i*stride+K, j*stride : j*stride+K]
               out[:, f, i, j] = sum of (patch * self.W[f]) over all axes + self.b[f]
               # np.sum(patch * self.W[f], axis=(1,2,3)) gives shape (N,)
        5. Cache x_padded and x.shape for the backward pass
        """
        # *** TODO ***
        batch_size, num_channels, height, width = x.shape

        # step 1 (set padding)
        if self.padding > 0: 
            self.x_padded = np.pad(x, ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)), mode='constant')
        else:
            self.x_padded = x
        
        # step 2 (compute height and width of output)
        H_out = (height + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (width + 2*self.padding - self.kernel_size) // self.stride + 1

        # step 3 (make output array)
        output = np.zeros((batch_size, self.num_filters, H_out, W_out))

        # step 4 (loop)
        for f in range(self.num_filters):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size
                    
                    patch = self.x_padded[:, :, h_start:h_end, w_start:w_end]
                    output[:, f, i, j] = np.sum(patch * self.W[f], axis=(1, 2, 3)) + self.b[f]

        # step 5 (cache shape for backward pass)
        self.x_shape = x.shape

        return output

    def backward(self, dout):
        """
        Given the gradient of the loss w.r.t. the output, compute gradients
        for the weights (dW), biases (db), and the layer input (dx).

        Parameters
        ----------
        dout : np.ndarray, shape (N, num_filters, H_out, W_out)

        Returns
        -------
        dx : np.ndarray, shape (N, C, H, W)

        Steps to implement
        ------------------
        1. Allocate: dW same shape as W, db same shape as b, dx_padded same shape as x_padded
        2. db = dout summed over axes (0, 2, 3)   → shape (num_filters,)
        3. Loop over filters f, rows i, columns j:
               patch   = x_padded[:, :, i*s:i*s+K, j*s:j*s+K]   shape (N,C,K,K)
               dout_ij = dout[:, f, i, j]                          shape (N,)
               dW[f]  += sum over N of (dout_ij[:,None,None,None] * patch)
               dx_padded[:, :, i*s:i*s+K, j*s:j*s+K] += dout_ij[:,None,None,None] * W[f]
        4. Strip padding from dx_padded to get dx (if padding > 0)
        """
        # *** TODO ***
        # step 1 (allocate)
        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)
        dx_padded = np.zeros(self.x_padded.shape)

        # step 2
        N, _, H_out, W_out = dout.shape
        db = np.sum(dout, axis=(0,2,3))

        # step 3
        for f in range(self.num_filters):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size
                    
                    patch = self.x_padded[:, :, h_start:h_end, w_start:w_end]
                    dout_ij = dout[:, f, i, j]
                    
                    dW[f] += np.sum(dout_ij[:, None, None, None] * patch, axis=0)
                    dx_padded[:, :, h_start:h_end, w_start:w_end] += dout_ij[:, None, None, None] * self.W[f]

        # step 4
        if self.padding > 0:
            p = self.padding
            dx = dx_padded[:, :, p:-p, p:-p]
        else:
            dx = dx_padded

        self.dW = dW
        self.db = db

        return dx

    # --- Pre-implemented: momentum update is the same for every layer ---
    def update(self, lr, momentum=0.9):
        """
        SGD with momentum.

        velocity = momentum * velocity - lr * gradient
        weight   = weight + velocity
        """
        self.vW = momentum * self.vW - lr * self.dW
        self.vb = momentum * self.vb - lr * self.db
        self.W += self.vW
        self.b += self.vb


# =============================================================================
# MAX POOLING LAYER  (pre-implemented)
# =============================================================================

class MaxPoolLayer:
    """
    Max Pooling: down-samples each feature map by keeping only the largest
    value in each non-overlapping window.

    Why? It makes the representation smaller and more robust to small shifts
    in the input (translation invariance).

    This is provided for you — focus your effort on the conv and FC layers.
    """

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride    = stride
        self.x         = None

    def forward(self, x):
        N, C, H, W = x.shape
        P, S       = self.pool_size, self.stride
        H_out      = (H - P) // S + 1
        W_out      = (W - P) // S + 1

        out = np.zeros((N, C, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                window     = x[:, :, i*S:i*S+P, j*S:j*S+P]
                out[:, :, i, j] = np.max(window, axis=(2, 3))

        self.x = x
        return out

    def backward(self, dout):
        x, P, S    = self.x, self.pool_size, self.stride
        N, C, H, W = x.shape
        H_out      = (H - P) // S + 1
        W_out      = (W - P) // S + 1
        dx         = np.zeros_like(x)

        for i in range(H_out):
            for j in range(W_out):
                window = x[:, :, i*S:i*S+P, j*S:j*S+P]              # (N,C,P,P)
                max_v  = np.max(window, axis=(2, 3), keepdims=True)   # (N,C,1,1)
                mask   = (window == max_v)                             # (N,C,P,P)
                # distribute gradient only to the position that was the max
                dx[:, :, i*S:i*S+P, j*S:j*S+P] += \
                    mask * dout[:, :, i, j][:, :, None, None]

        return dx


# =============================================================================
# FULLY CONNECTED LAYER
# =============================================================================

class FCLayer:
    """
    A standard fully-connected (dense) layer: out = x @ W + b

    Every input neuron is connected to every output neuron.
    This is identical to the layers inside an ANN.
    """

    def __init__(self, in_features, out_features):
        self.in_features  = in_features
        self.out_features = out_features

        # He initialisation
        self.W  = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b  = np.zeros(out_features)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

        self.x = None   # cached input for backward pass

    def forward(self, x):
        """
        Compute the linear transformation: out = x @ W + b

        Parameters
        ----------
        x : np.ndarray, shape (N, in_features)

        Returns
        -------
        out : np.ndarray, shape (N, out_features)

        Steps to implement
        ------------------
        1. Cache x  (you will need it in backward)
        2. Return x @ self.W + self.b
        """
        # *** TODO ***
        self.x = x
        out = x @ self.W + self.b
        return out

    def backward(self, dout):
        """
        Compute gradients w.r.t. weights, biases, and the input.

        Parameters
        ----------
        dout : np.ndarray, shape (N, out_features) — gradient from next layer

        Returns
        -------
        dx : np.ndarray, shape (N, in_features)

        Steps to implement
        ------------------
        dW = self.x.T  @  dout          shape: (in_features, out_features)
        db = np.sum(dout, axis=0)        shape: (out_features,)
        dx = dout  @  self.W.T           shape: (N, in_features)

        Store dW and db as self.dW and self.db, then return dx.
        """
        # *** TODO ***
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis = 0)
        
        return dout @ self.W.T

    # --- Pre-implemented ---
    def update(self, lr, momentum=0.9):
        self.vW = momentum * self.vW - lr * self.dW
        self.vb = momentum * self.vb - lr * self.db
        self.W += self.vW
        self.b += self.vb


# =============================================================================
# CNN — wires all layers together
# =============================================================================

class CNN:
    """
    The full network. Your job here is to:
        1. Create the layer objects in __init__
        2. Call them in the right order in forward
        3. Call them in reverse order in backward
        4. Trigger parameter updates in update
    """

    def __init__(self):
        """
        Instantiate the layers below using the architecture from the header.

        *** TODO ***
        Replace each None with the correct layer constructor call:
            ConvLayer(in_channels, num_filters, kernel_size)
            MaxPoolLayer(pool_size, stride)
            FCLayer(in_features, out_features)

        Architecture reminder:
            conv1  — ConvLayer: 1 in_channel,  8 filters, kernel 3
            pool1  — MaxPoolLayer: pool 2, stride 2
            conv2  — ConvLayer: 8 in_channels, 16 filters, kernel 3
            pool2  — MaxPoolLayer: pool 2, stride 2
            fc1    — FCLayer: 400 → 128
            fc2    — FCLayer: 128 → 10
        """
        self.conv1 = ConvLayer(1,8,3)   # *** TODO ***
        self.pool1 = MaxPoolLayer(2, 2)   # *** TODO ***
        self.conv2 = ConvLayer(8, 16, 3)   # *** TODO ***
        self.pool2 = MaxPoolLayer(2,2)   # *** TODO ***
        self.fc1   = FCLayer(400,128)   # *** TODO ***
        self.fc2   = FCLayer(128, 10)  # *** TODO ***

        # These store intermediate values needed for backprop
        self.relu1_input = None
        self.relu2_input = None
        self.relu3_input = None
        self.flat_shape  = None

    def forward(self, x):
        """
        Pass data through every layer from input to output.

        Parameters
        ----------
        x : np.ndarray, shape (N, 1, 28, 28)

        Returns
        -------
        probs : np.ndarray, shape (N, 10) — class probabilities

        *** TODO ***
        Follow this exact sequence (each arrow is one line of code):

            x  → conv1  → save as relu1_input → relu → pool1  → x
            x  → conv2  → save as relu2_input → relu → pool2  → x
            x  → flatten (x.reshape(N, -1), save original shape) → x
            x  → fc1    → save as relu3_input → relu            → x
            x  → fc2    → softmax                               → probs

        Tip: save x BEFORE the relu call, e.g.:
            self.relu1_input = x          # save
            x = relu(x)                   # then activate
        """
        # *** TODO ***
        # line 1
        x = self.conv1.forward(x)
        self.relu1_input = x
        x = relu(self.relu1_input)
        x = self.pool1.forward(x)

        # line 2
        x = self.conv2.forward(x)
        self.relu2_input = x
        x = relu(self.relu2_input)
        x = self.pool2.forward(x)

        # line 3
        self.flat_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        # line 4
        x = self.fc1.forward(x)
        self.relu3_input = x
        x = relu(self.relu3_input)

        # line 5
        x = self.fc2.forward(x)
        probs = softmax(x)

        return probs



    def backward(self, grad):
        """
        Send the loss gradient backwards through every layer (reverse order).

        Parameters
        ----------
        grad : np.ndarray, shape (N, 10) — gradient from cross_entropy_loss

        *** TODO ***
        Reverse sequence:

            grad → fc2.backward    → grad
            grad → relu_backward(grad, relu3_input) → grad
            grad → fc1.backward    → grad
            grad → reshape back to self.flat_shape  → grad
            grad → pool2.backward  → grad
            grad → relu_backward(grad, relu2_input) → grad
            grad → conv2.backward  → grad
            grad → pool1.backward  → grad
            grad → relu_backward(grad, relu1_input) → grad
            grad → conv1.backward  → (grad not used further)
        """
        # *** TODO ***
        # line 1
        grad = self.fc2.backward(grad)

        # line 2
        grad = relu_backward(grad,self.relu3_input)

        # line 3
        grad = self.fc1.backward(grad)

        # line 4
        grad = grad.reshape(self.flat_shape)

        # line 5
        grad = self.pool2.backward(grad)

        # line 6
        grad = relu_backward(grad, self.relu2_input)

        # line 7
        grad = self.conv2.backward(grad)

        # line 8
        grad = self.pool1.backward(grad)

        # line 9
        grad = relu_backward(grad, self.relu1_input)

        # line 10
        grad = self.conv1.backward(grad)

        return grad


        

    def update(self, lr, momentum=0.9):
        """
        Update every layer that has learnable parameters.

        *** TODO ***
        Call .update(lr, momentum) on: conv1, conv2, fc1, fc2
        (MaxPoolLayer has no weights, so skip pool1 and pool2)
        """
        # *** TODO ***
        self.conv1.update(lr, momentum)
        self.conv2.update(lr, momentum)
        self.fc1.update(lr, momentum)
        self.fc2.update(lr, momentum)

    # --- Pre-implemented ---
    def predict(self, x):
        """Run a forward pass and return the predicted class index for each image."""
        probs = self.forward(x)
        return np.argmax(probs, axis=1)
