"""
compare_results.py
==================
Side-by-side comparison of CNN vs ANN training results.

Run this AFTER both models have been trained:
    cd cnn && python train_cnn.py
    cd ann && python train_ann.py
    cd ..  && python compare_results.py

Reads:
    cnn/results/cnn_results.csv
    ann/results/ann_results.csv

Outputs:
    comparison_plots.png  — saved to the project root
"""

import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# PATHS
# =============================================================================

CNN_CSV    = os.path.join('cnn', 'results', 'cnn_results.csv')
ANN_CSV    = os.path.join('ann', 'results', 'ann_results.csv')
OUTPUT_PNG = 'comparison_plots.png'

# =============================================================================
# COLOURS
# =============================================================================

CNN_COLOR = '#2563EB'   # blue
ANN_COLOR = '#DC2626'   # red

# =============================================================================
# LOAD DATA
# =============================================================================

def load_csv(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\nCould not find: {filepath}\n"
            f"Make sure you have trained that model first."
        )
    return pd.read_csv(filepath)


def print_summary(cnn_df, ann_df):
    """Print a plain-text per-epoch table to the console."""
    epochs = min(len(cnn_df), len(ann_df))
    header = (f"{'Epoch':>6}  {'CNN Val Loss':>13}  {'CNN Val Acc':>11}  "
              f"{'ANN Val Loss':>13}  {'ANN Val Acc':>11}  {'Winner':>7}")
    print("\n" + "=" * len(header))
    print("  CNN vs ANN — Per-Epoch Validation Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for i in range(epochs):
        c = cnn_df.iloc[i]
        a = ann_df.iloc[i]
        if   c['val_accuracy'] > a['val_accuracy'] + 0.0005: winner = 'CNN'
        elif a['val_accuracy'] > c['val_accuracy'] + 0.0005: winner = 'ANN'
        else:                                                  winner = 'tie'
        print(f"{int(c['epoch']):>6}  {c['val_loss']:>13.6f}  {c['val_accuracy']:>11.4f}  "
              f"{a['val_loss']:>13.6f}  {a['val_accuracy']:>11.4f}  {winner:>7}")

    print("-" * len(header))
    best_cnn = cnn_df.loc[cnn_df['val_accuracy'].idxmax()]
    best_ann = ann_df.loc[ann_df['val_accuracy'].idxmax()]
    print(f"\n  Best CNN val accuracy : {best_cnn['val_accuracy']:.4f}  "
          f"(epoch {int(best_cnn['epoch'])})")
    print(f"  Best ANN val accuracy : {best_ann['val_accuracy']:.4f}  "
          f"(epoch {int(best_ann['epoch'])})")
    diff = best_cnn['val_accuracy'] - best_ann['val_accuracy']
    if   diff > 0.0005:  print(f"\n  CNN outperformed ANN by {diff*100:.2f} pp")
    elif diff < -0.0005: print(f"\n  ANN outperformed CNN by {abs(diff)*100:.2f} pp")
    else:                print(f"\n  Both models tied on best validation accuracy")
    print()


# =============================================================================
# PLOTTING
# =============================================================================

def make_plots(cnn_df, ann_df):
    epochs     = np.arange(1, min(len(cnn_df), len(ann_df)) + 1)
    cnn_df     = cnn_df.iloc[:len(epochs)]
    ann_df     = ann_df.iloc[:len(epochs)]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('CNN vs ANN — MNIST Training Comparison', fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

    # ── 1. Validation Accuracy ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])   # full-width top row
    ax1.plot(epochs, cnn_df['val_accuracy'], color=CNN_COLOR, marker='o',
             linewidth=2, markersize=5, label='CNN')
    ax1.plot(epochs, ann_df['val_accuracy'], color=ANN_COLOR, marker='s',
             linewidth=2, markersize=5, label='ANN', linestyle='--')
    ax1.set_title('Validation Accuracy over Epochs', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(epochs)
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    # annotate best points
    for df, color, marker in [(cnn_df, CNN_COLOR, 'o'), (ann_df, ANN_COLOR, 's')]:
        best_idx = df['val_accuracy'].idxmax()
        best_ep  = int(df.loc[best_idx, 'epoch'])
        best_acc = df.loc[best_idx, 'val_accuracy']
        ax1.annotate(f'{best_acc:.4f}',
                     xy=(best_ep, best_acc),
                     xytext=(best_ep, best_acc + 0.03),
                     ha='center', fontsize=9, color=color,
                     arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    # ── 2. Training Loss ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(epochs, cnn_df['train_loss'], color=CNN_COLOR, marker='o',
             linewidth=2, markersize=4, label='CNN')
    ax2.plot(epochs, ann_df['train_loss'], color=ANN_COLOR, marker='s',
             linewidth=2, markersize=4, label='ANN', linestyle='--')
    ax2.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.set_xticks(epochs)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ── 3. Validation Loss ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(epochs, cnn_df['val_loss'], color=CNN_COLOR, marker='o',
             linewidth=2, markersize=4, label='CNN')
    ax3.plot(epochs, ann_df['val_loss'], color=ANN_COLOR, marker='s',
             linewidth=2, markersize=4, label='ANN', linestyle='--')
    ax3.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Cross-Entropy Loss')
    ax3.set_xticks(epochs)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ── 4. Training Accuracy ──────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(epochs, cnn_df['train_accuracy'], color=CNN_COLOR, marker='o',
             linewidth=2, markersize=4, label='CNN')
    ax4.plot(epochs, ann_df['train_accuracy'], color=ANN_COLOR, marker='s',
             linewidth=2, markersize=4, label='ANN', linestyle='--')
    ax4.set_title('Training Accuracy', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_xticks(epochs)
    ax4.set_ylim(0, 1.05)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # ── 5. Train/Val Accuracy Gap (overfitting indicator) ─────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    cnn_gap = cnn_df['train_accuracy'].values - cnn_df['val_accuracy'].values
    ann_gap = ann_df['train_accuracy'].values - ann_df['val_accuracy'].values
    ax5.bar(epochs - 0.2, cnn_gap, width=0.35, color=CNN_COLOR, alpha=0.75, label='CNN')
    ax5.bar(epochs + 0.2, ann_gap, width=0.35, color=ANN_COLOR, alpha=0.75, label='ANN')
    ax5.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax5.set_title('Train − Val Accuracy Gap\n(higher = more overfitting)',
                  fontsize=12, fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Accuracy Gap')
    ax5.set_xticks(epochs)
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')

    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    print(f"Plots saved -> {OUTPUT_PNG}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading results...")
    cnn_df = load_csv(CNN_CSV)
    ann_df = load_csv(ANN_CSV)

    print_summary(cnn_df, ann_df)
    make_plots(cnn_df, ann_df)

    print("\nReflection Questions")
    print("─" * 50)
    questions = [
        "1. Which model achieved higher validation accuracy? By how much?",
        "2. The CNN has far fewer early-layer parameters than the ANN\n"
        "   (which connects all 784 pixels directly). Why might it still\n"
        "   perform better despite fewer connections?",
        "3. Look at the Train-Val Accuracy Gap chart. Which model shows\n"
        "   more overfitting? What does that tell you?",
        "4. Did either model's val_loss increase while train_loss kept\n"
        "   falling? What is that called and why does it happen?",
        "5. How might increasing the number of epochs or adjusting the\n"
        "   learning rate affect each model differently?",
    ]
    for q in questions:
        print(f"\n  {q}")
    print()


if __name__ == '__main__':
    main()
