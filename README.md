# ML Models from Scratch

Polynomial Regression and Multi-Class Logistic Regression built from scratch using NumPy only. No sklearn model classes used.

---

## Files

| File | Model | Dataset |
|---|---|---|
| `polynomial_regression.py` | Polynomial Regression | Diabetes (sklearn) |
| `logistic_regression.py` | Multi-Class Logistic Regression | Iris (sklearn) |

---

## How to Run

```bash
pip install numpy matplotlib scikit-learn

python polynomial_regression.py
python logistic_regression.py
```

---

## How It Works

### Polynomial Regression

Expands a single input feature into polynomial terms `[1, x, x², x³]` so a linear model can fit curved data.

**Loss (MSE + L2 regularisation):**
```
Loss = mean((ŷ - y)²)  +  λ × sum(w²)
```

**Weight update (gradient descent):**
```
w = w - lr × gradient
```

The λ term shrinks weights to prevent overfitting. Bias is not penalised.

---

### Multi-Class Logistic Regression (Softmax)

Uses softmax to produce probabilities over 3 classes. Learns a weight matrix (one column per class).

**Forward pass:**
```
Z     = X @ W        (raw scores)
probs = softmax(Z)   (probabilities)
```

**Loss (Cross-Entropy + L2):**
```
Loss = -mean(Σ y × log(ŷ))  +  λ × sum(w²)
```

---

### K-Fold Cross-Validation

1. Shuffle and split data into 5 equal folds
2. For each fold: train on 4 folds, validate on the remaining 1
3. Rotate through all 5 folds
4. Average the errors — gives a reliable estimate of real-world performance

---

## Results

**Polynomial Regression (Diabetes dataset):**
- Avg Train MSE: `0.0382`
- Avg Val MSE: `0.0382`
- Test MSE: `0.0382`

**Logistic Regression (Iris dataset):**
- Avg Train Accuracy: `~96%`
- Test Accuracy: `96.67%`
