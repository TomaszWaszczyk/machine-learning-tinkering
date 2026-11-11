"""Lesson 3 - several features, and how to tell whether the model is any good.

Run it with:

    python 03_multiple_features.py

Real problems rarely have a single input. Here we predict the price of an
apartment from four features. The model barely changes:

    price = w1*size + w2*rooms + w3*age + w4*km_to_centre + b

New ideas in this lesson:

  * train / test split - judge the model on data it has never seen
  * feature scaling    - why gradient descent needs comparable feature ranges
  * R^2 and RMSE       - two numbers that describe the quality of a fit
"""

import numpy as np

from datasets import (
    APARTMENT_FEATURES,
    APARTMENT_TRUE_BIAS,
    APARTMENT_TRUE_WEIGHTS,
    make_apartment_data,
)
from plotting import BLUE, MUTED, plt, save

rows, prices = make_apartment_data()
X = np.array(rows)  # shape (300, 4): one row per apartment, one column per feature
y = np.array(prices)  # shape (300,)

# --- Train / test split -----------------------------------------------------
# A model can always look good on the examples it was fitted to. The honest
# question is how it does on NEW apartments, so we hide 20% of the data and
# only look at it at the very end.
rng = np.random.default_rng(0)
order = rng.permutation(len(X))
n_test = len(X) // 5
test, train = order[:n_test], order[n_test:]
X_train, y_train = X[train], y[train]
X_test, y_test = X[test], y[test]
print(f"{len(X_train)} training examples, {len(X_test)} test examples\n")


def gradient_descent(X, y, learning_rate, epochs):
    """Same algorithm as before; w is now a vector with one weight per feature."""
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        error = X @ w + b - y
        w -= learning_rate * 2 * (X.T @ error) / len(y)
        b -= learning_rate * 2 * np.mean(error)
    return w, b


def report(name, y_true, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    # R^2: which fraction of the variation in prices does the model explain?
    # 1.0 is perfect, 0.0 is no better than always guessing the average price.
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    print(f"{name:<6} RMSE = {rmse:6.1f}k PLN    R^2 = {r2:.4f}")


# --- Attempt 1: raw features -------------------------------------------------
# size_m2 goes up to 120 while rooms goes up to about 5. The loss surface is
# then a long, narrow valley: any learning rate big enough to make progress
# along the valley overshoots across it, and the numbers blow up.
print("gradient descent on RAW features, learning rate 0.01:")
with np.errstate(over="ignore", invalid="ignore"):
    w_raw, b_raw = gradient_descent(X_train, y_train, learning_rate=0.01, epochs=1000)
print(f"  weights = {w_raw}   <- diverged\n")

# --- Attempt 2: standardised features ---------------------------------------
# Rescale every column to mean 0 and standard deviation 1. Note that the mean
# and std come from the TRAINING set only and are reused for the test set:
# the test set must not influence anything the model learns.
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_s = (X_train - mean) / std
X_test_s = (X_test - mean) / std

w_s, b_s = gradient_descent(X_train_s, y_train, learning_rate=0.05, epochs=2000)

print("gradient descent on STANDARDISED features, learning rate 0.05:")
report("train", y_train, X_train_s @ w_s + b_s)
report("test", y_test, X_test_s @ w_s + b_s)

# --- Reading the model -------------------------------------------------------
# Convert the weights back to the original units so we can interpret them:
# "one extra square metre adds about 9k PLN, all else being equal".
w = w_s / std
b = b_s - np.sum(w_s * mean / std)

print("\nwhat the model learned (original units):")
print(f"  {'feature':<14}{'learned':>10}{'true':>10}")
for name, learned, true in zip(APARTMENT_FEATURES, w, APARTMENT_TRUE_WEIGHTS):
    print(f"  {name:<14}{learned:>10.2f}{true:>10.2f}")
print(f"  {'bias':<14}{b:>10.2f}{APARTMENT_TRUE_BIAS:>10.2f}")

# On standardised features the size of a weight tells you how much that
# feature matters, because all features now share the same scale.
print("\nweights on standardised features (bigger magnitude = more influence):")
for name, weight in sorted(zip(APARTMENT_FEATURES, w_s), key=lambda p: -abs(p[1])):
    print(f"  {name:<14}{weight:>10.1f}")

example = np.array([60, 3, 10, 4])
print(f"\na 60 m2, 3-room, 10-year-old flat 4 km from the centre: {example @ w + b:.0f}k PLN")

# --- Figure ------------------------------------------------------------------
y_pred = X_test_s @ w_s + b_s
lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())

fig, ax = plt.subplots(figsize=(5.4, 5.2))
ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.2, linestyle="--")
ax.scatter(y_test, y_pred, s=28, color=BLUE, edgecolor="white", linewidth=0.8)
ax.text(
    0.97,
    0.04,
    "dashed line = perfect prediction",
    transform=ax.transAxes,
    ha="right",
    color=MUTED,
    fontsize=9,
)
ax.set_title("Test set: predicted vs actual price")
ax.set_xlabel("actual price (k PLN)")
ax.set_ylabel("predicted price (k PLN)")
ax.set_aspect("equal")
save(fig, "predicted_vs_actual.png")
