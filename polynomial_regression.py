"""
Polynomial Regression from Scratch
Dataset: Diabetes (sklearn built-in) — 442 samples, predicts disease progression
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# ── Load Dataset ──────────────────────────────────────────────
X_all, y_all = load_diabetes(return_X_y=True)

# Use only 1 feature (BMI) to keep polynomial expansion simple
X_all = X_all[:, 2].reshape(-1, 1)

# Normalise both X and y to [0, 1] range
X_all = (X_all - X_all.min()) / (X_all.max() - X_all.min())
y_all = (y_all - y_all.min()) / (y_all.max() - y_all.min())

# ── Train / Test Split (manual) ───────────────────────────────
np.random.seed(42)
idx       = np.random.permutation(len(y_all))
split     = int(0.8 * len(y_all))
X_train, y_train = X_all[idx[:split]], y_all[idx[:split]]
X_test,  y_test  = X_all[idx[split:]], y_all[idx[split:]]

# ── Polynomial Feature Expansion ─────────────────────────────
def poly_features(X, degree):
    # Builds [1, x, x^2, x^3, ...] for each sample
    cols = [np.ones(len(X))]
    for d in range(1, degree + 1):
        cols.append(X[:, 0] ** d)
    return np.column_stack(cols)

# ── Train with Gradient Descent + L2 Regularisation ──────────
def train(X, y, degree=3, lr=0.1, epochs=1000, lam=0.1):
    Xp = poly_features(X, degree)
    n, d = Xp.shape
    w = np.zeros(d)
    losses = []

    for _ in range(epochs):
        pred  = Xp @ w
        error = pred - y

        # Gradient of MSE + L2 penalty (bias term w[0] not penalised)
        grad    = (Xp.T @ error) / n
        reg     = lam * w / n
        reg[0]  = 0
        w      -= lr * (grad + reg)

        loss = np.mean(error**2) + (lam / (2*n)) * np.sum(w[1:]**2)
        losses.append(loss)

    return w, losses

def predict(X, w, degree=3):
    return poly_features(X, degree) @ w

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# ── K-Fold Cross Validation (manual) ─────────────────────────
K = 5
fold_size = len(X_train) // K
idx_cv    = np.random.permutation(len(X_train))

train_errors = []
val_errors   = []

print("K-Fold Cross Validation\n" + "-"*35)
for k in range(K):
    val_idx   = idx_cv[k*fold_size : (k+1)*fold_size]
    train_idx = np.concatenate([idx_cv[:k*fold_size], idx_cv[(k+1)*fold_size:]])

    Xtr, ytr = X_train[train_idx], y_train[train_idx]
    Xvl, yvl = X_train[val_idx],   y_train[val_idx]

    w, _ = train(Xtr, ytr)

    tr_err = mse(ytr, predict(Xtr, w))
    vl_err = mse(yvl, predict(Xvl, w))
    train_errors.append(tr_err)
    val_errors.append(vl_err)
    print(f"Fold {k+1}: Train MSE = {tr_err:.4f}  |  Val MSE = {vl_err:.4f}")

# ── Final Model on full training set ─────────────────────────
w_final, loss_curve = train(X_train, y_train)
test_mse = mse(y_test, predict(X_test, w_final))

print(f"\nAvg Train MSE : {np.mean(train_errors):.4f}")
print(f"Avg Val   MSE : {np.mean(val_errors):.4f}")
print(f"Test MSE      : {test_mse:.4f}")

# ── Plots ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Training vs Validation Error per fold
axes[0].plot(range(1, K+1), train_errors, 'o-', color='steelblue', label='Train MSE')
axes[0].plot(range(1, K+1), val_errors,   's-', color='tomato',    label='Val MSE')
axes[0].set_title('Training vs Validation Error')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('MSE')
axes[0].legend()

# 2. Loss convergence curve
axes[1].plot(loss_curve, color='darkorange')
axes[1].set_title('Loss Curve (Gradient Descent)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')

# 3. Polynomial fit on data
x_line = np.linspace(0, 1, 200).reshape(-1, 1)
y_line = predict(x_line, w_final)
axes[2].scatter(X_train, y_train, s=10, alpha=0.4, color='steelblue', label='Train')
axes[2].scatter(X_test,  y_test,  s=10, alpha=0.4, color='tomato',    label='Test')
axes[2].plot(x_line, y_line, color='black', linewidth=2, label='Poly Fit')
axes[2].set_title('Polynomial Regression Fit')
axes[2].set_xlabel('BMI (normalised)')
axes[2].set_ylabel('Target (normalised)')
axes[2].legend()

plt.tight_layout()
plt.savefig('regression_results.png', dpi=150)
plt.show()
print("\nPlot saved → regression_results.png")
