"""
Multi-Class Logistic Regression from Scratch
Dataset: Iris (sklearn built-in) — 150 samples, 3 flower species, 4 features
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ── Load Dataset ──────────────────────────────────────────────
X_all, y_all = load_iris(return_X_y=True)

# Normalise features to zero mean and unit variance
X_all = (X_all - X_all.mean(axis=0)) / X_all.std(axis=0)

# ── Train / Test Split (manual) ───────────────────────────────
np.random.seed(42)
idx   = np.random.permutation(len(y_all))
split = int(0.8 * len(y_all))
X_train, y_train = X_all[idx[:split]], y_all[idx[:split]]
X_test,  y_test  = X_all[idx[split:]], y_all[idx[split:]]

# ── Helper Functions ──────────────────────────────────────────
def softmax(Z):
    # Subtract row max for numerical stability
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)

def one_hot(y, n_classes):
    oh = np.zeros((len(y), n_classes))
    oh[np.arange(len(y)), y] = 1
    return oh

# ── Train with Gradient Descent + L2 Regularisation ──────────
def train(X, y, lr=0.1, epochs=1000, lam=0.01):
    n, d      = X.shape
    n_classes = len(np.unique(y))

    # Add bias column to X
    Xb = np.hstack([X, np.ones((n, 1))])     # shape: (n, d+1)
    W  = np.zeros((d + 1, n_classes))         # weight matrix
    Y  = one_hot(y, n_classes)
    losses = []

    for _ in range(epochs):
        probs = softmax(Xb @ W)               # forward pass

        # Cross-entropy loss + L2 penalty (bias row W[-1] not penalised)
        ce_loss = -np.mean(np.sum(Y * np.log(probs + 1e-15), axis=1))
        reg_loss = (lam / (2*n)) * np.sum(W[:-1]**2)
        losses.append(ce_loss + reg_loss)

        # Gradient
        grad        = (Xb.T @ (probs - Y)) / n
        reg         = (lam / n) * W.copy()
        reg[-1]     = 0                        # no penalty on bias
        W          -= lr * (grad + reg)

    return W, losses

def predict(X, W):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    return np.argmax(softmax(Xb @ W), axis=1)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def cross_entropy(X, y, W):
    Xb    = np.hstack([X, np.ones((len(X), 1))])
    probs = softmax(Xb @ W)
    Y     = one_hot(y, W.shape[1])
    return -np.mean(np.sum(Y * np.log(probs + 1e-15), axis=1))

# ── K-Fold Cross Validation (manual) ─────────────────────────
K = 5
fold_size = len(X_train) // K
idx_cv    = np.random.permutation(len(X_train))

train_losses = []
val_losses   = []

print("K-Fold Cross Validation\n" + "-"*45)
for k in range(K):
    val_idx   = idx_cv[k*fold_size : (k+1)*fold_size]
    train_idx = np.concatenate([idx_cv[:k*fold_size], idx_cv[(k+1)*fold_size:]])

    Xtr, ytr = X_train[train_idx], y_train[train_idx]
    Xvl, yvl = X_train[val_idx],   y_train[val_idx]

    W, _ = train(Xtr, ytr)

    tl = cross_entropy(Xtr, ytr, W)
    vl = cross_entropy(Xvl, yvl, W)
    ta = accuracy(ytr, predict(Xtr, W))
    va = accuracy(yvl, predict(Xvl, W))

    train_losses.append(tl)
    val_losses.append(vl)
    print(f"Fold {k+1}: Train Loss={tl:.4f} Acc={ta:.2%}  |  Val Loss={vl:.4f} Acc={va:.2%}")

# ── Final Model on full training set ─────────────────────────
W_final, loss_curve = train(X_train, y_train)
test_loss = cross_entropy(X_test, y_test, W_final)
test_acc  = accuracy(y_test, predict(X_test, W_final))

print(f"\nAvg Train Loss : {np.mean(train_losses):.4f}")
print(f"Avg Val   Loss : {np.mean(val_losses):.4f}")
print(f"Test Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {test_acc:.2%}")

# ── Plots ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Training vs Validation Loss per fold
axes[0].plot(range(1, K+1), train_losses, 'o-', color='steelblue', label='Train Loss')
axes[0].plot(range(1, K+1), val_losses,   's-', color='tomato',    label='Val Loss')
axes[0].set_title('Training vs Validation Error')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].legend()

# 2. Loss convergence curve
axes[1].plot(loss_curve, color='darkorange')
axes[1].set_title('Loss Curve (Gradient Descent)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')

# 3. Scatter plot of test predictions (2 features)
colors = ['tomato', 'steelblue', 'mediumseagreen']
labels = ['Setosa', 'Versicolor', 'Virginica']
preds  = predict(X_test, W_final)
for cls in range(3):
    mask = y_test == cls
    axes[2].scatter(X_test[mask, 0], X_test[mask, 1],
                    color=colors[cls], label=labels[cls], s=60,
                    marker='o' if (preds[mask] == cls).all() else 'x')
axes[2].set_title('Test Set — Sepal Features')
axes[2].set_xlabel('Sepal Length (normalised)')
axes[2].set_ylabel('Sepal Width (normalised)')
axes[2].legend()

plt.tight_layout()
plt.savefig('classification_results.png', dpi=150)
plt.show()
print("\nPlot saved → classification_results.png")
