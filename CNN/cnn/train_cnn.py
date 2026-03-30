"""
train_cnn.py
============
Training Script for the CNN on MNIST

Trains the CNN defined in cnn_model.py using mini-batch SGD with momentum,
then saves per-epoch metrics to results/cnn_results.csv.

No external libraries — numpy and Python standard library only.
"""

import numpy as np
import csv
import os
import time

from cnn_model import CNN, cross_entropy_loss

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

LEARNING_RATE = 0.01
MOMENTUM      = 0.9
BATCH_SIZE    = 32
EPOCHS        = 10

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, '../data')
RESULTS_FILE = 'results/cnn_results.csv'


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_dir):
    """Load the four .npy files and split train_val 90/10."""
    train_val_images = np.load(os.path.join(data_dir, 'train_images.npy'))
    train_val_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
    test_images      = np.load(os.path.join(data_dir, 'test_images.npy'))
    test_labels      = np.load(os.path.join(data_dir, 'test_labels.npy'))

    split        = int(train_val_images.shape[0] * 0.9)
    train_images = train_val_images[:split]
    train_labels = train_val_labels[:split]
    val_images   = train_val_images[split:]
    val_labels   = train_val_labels[split:]

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess(images):
    """
    Normalise to [0, 1] and add a channel dimension.

    The CNN expects channels-first format: (N, 1, 28, 28).
    The channel dimension (size 1) represents grayscale.
    """
    images = images.astype(np.float64) / 255.0
    images = images[:, np.newaxis, :, :]   # (N, 28, 28) -> (N, 1, 28, 28)
    return images


# =============================================================================
# ACCURACY
# =============================================================================

def compute_accuracy(predictions, labels):
    return np.sum(predictions == labels) / len(labels)


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(model, images, labels, batch_size, lr, momentum):
    """
    One full pass over the training set using mini-batch SGD.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    n_samples   = images.shape[0]
    indices     = np.random.permutation(n_samples)
    num_batches = int(np.ceil(n_samples / batch_size))
    total_loss  = 0.0
    all_preds   = []

    for i in range(num_batches):
        batch_idx    = indices[i * batch_size : (i + 1) * batch_size]
        batch_images = images[batch_idx]
        batch_labels = labels[batch_idx]

        probs        = model.forward(batch_images)
        loss, grad   = cross_entropy_loss(probs, batch_labels)

        model.backward(grad)
        model.update(lr, momentum)

        total_loss += loss
        all_preds.append(np.argmax(probs, axis=1))

    avg_loss = total_loss / num_batches
    accuracy = compute_accuracy(np.concatenate(all_preds), labels)
    return avg_loss, accuracy


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, images, labels, batch_size):
    """
    Measure loss and accuracy on a dataset without updating weights.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    n_samples   = images.shape[0]
    num_batches = int(np.ceil(n_samples / batch_size))
    total_loss  = 0.0
    all_preds   = []

    for i in range(num_batches):
        batch_images = images[i * batch_size : (i + 1) * batch_size]
        batch_labels = labels[i * batch_size : (i + 1) * batch_size]

        probs      = model.forward(batch_images)
        loss, _    = cross_entropy_loss(probs, batch_labels)

        total_loss += loss
        all_preds.append(np.argmax(probs, axis=1))

    avg_loss = total_loss / num_batches
    accuracy = compute_accuracy(np.concatenate(all_preds), labels)
    return avg_loss, accuracy


# =============================================================================
# RESULTS SAVING
# =============================================================================

def save_results(results, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading data...")
    train_images, train_labels, val_images, val_labels, test_images, test_labels = \
        load_data(DATA_DIR)

    print("Preprocessing...")
    train_images = preprocess(train_images)
    val_images   = preprocess(val_images)
    test_images  = preprocess(test_images)

    print(f"  Train : {train_images.shape}")
    print(f"  Val   : {val_images.shape}")
    print(f"  Test  : {test_images.shape}")

    model   = CNN()
    results = []

    print(f"\nTraining CNN — {EPOCHS} epochs | batch={BATCH_SIZE} | lr={LEARNING_RATE}\n")
    header = (f"{'Epoch':>6}  {'Train Loss':>12}  {'Train Acc':>10}  "
              f"{'Val Loss':>10}  {'Val Acc':>8}  {'Time':>7}")
    print(header)
    print("-" * len(header))

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_images, train_labels, BATCH_SIZE, LEARNING_RATE, MOMENTUM
        )
        val_loss, val_acc = evaluate(model, val_images, val_labels, BATCH_SIZE)

        elapsed = time.time() - t0
        print(f"{epoch:>6}  {train_loss:>12.6f}  {train_acc:>10.4f}  "
              f"{val_loss:>10.6f}  {val_acc:>8.4f}  {elapsed:>6.1f}s")

        results.append({
            'epoch':          epoch,
            'train_loss':     round(train_loss, 6),
            'train_accuracy': round(train_acc,  4),
            'val_loss':       round(val_loss,   6),
            'val_accuracy':   round(val_acc,    4),
        })

    print("\nEvaluating on test set...")
    test_loss, test_acc = evaluate(model, test_images, test_labels, BATCH_SIZE)
    print(f"  Test Loss     : {test_loss:.6f}")
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc * 100:.2f} %)")

    save_results(results, RESULTS_FILE)
    print(f"\nResults saved -> {RESULTS_FILE}")


if __name__ == '__main__':
    main()
