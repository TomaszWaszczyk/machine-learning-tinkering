"""Lesson 2 - the same model with numpy, and a formula that skips training.

Run it with:

    python 02_numpy_and_normal_equation.py

Two new ideas:

  * Vectorisation. Instead of looping over examples we write the maths on
    whole arrays. Same result as lesson 1, far less code, much faster.

  * The normal equation. Linear regression is special: the best parameters
    can be computed directly, with no iterations. Most ML models have no such
    shortcut, which is why gradient descent is still the thing to learn.

The script also draws the figures used in the README.
"""

import numpy as np

from datasets import make_salary_data
from plotting import BLUE, MUTED, ORANGE, plt, save

xs, ys = make_salary_data()
x = np.array(xs)  # shape (50,)
y = np.array(ys)  # shape (50,)


# --- Gradient descent, vectorised -------------------------------------------
def train(x, y, learning_rate=0.01, epochs=2000):
    w, b = 0.0, 0.0
    losses = []
    snapshots = {}  # (w, b) at a few moments, to draw the line "learning"
    for epoch in range(epochs + 1):
        error = w * x + b - y  # all 50 errors at once
        losses.append(np.mean(error**2))
        if epoch in (0, 1, 100, epochs):
            snapshots[epoch] = (w, b)
        w -= learning_rate * 2 * np.mean(error * x)
        b -= learning_rate * 2 * np.mean(error)
    return w, b, losses, snapshots


w_gd, b_gd, losses, snapshots = train(x, y)

# --- The normal equation ----------------------------------------------------
# Trick: glue a column of ones onto x. Then  w*x + b  becomes a single matrix
# product  X @ theta  with theta = [w, b], and the bias is "just another
# weight" that always multiplies 1.
X = np.column_stack([x, np.ones_like(x)])  # shape (50, 2)

# Setting the gradient of the loss to zero and solving gives
#     theta = (X^T X)^-1 X^T y
# We ask numpy to solve the linear system rather than invert the matrix,
# which is the numerically safer way to evaluate the same formula.
theta = np.linalg.solve(X.T @ X, X.T @ y)
w_ne, b_ne = theta

print("gradient descent :  w = %.4f   b = %.4f" % (w_gd, b_gd))
print("normal equation  :  w = %.4f   b = %.4f" % (w_ne, b_ne))
print("loss at the optimum: %.4f" % np.mean((X @ theta - y) ** 2))

# --- Figures ----------------------------------------------------------------
line_x = np.array([0, 10])

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(x, y, s=28, color=BLUE, edgecolor="white", linewidth=0.8, label="training data")
ax.plot(line_x, w_ne * line_x + b_ne, color=ORANGE, label="fitted line")
ax.set_title(f"The fitted model: salary = {w_ne:.2f} × years + {b_ne:.2f}")
ax.set_xlabel("years of experience")
ax.set_ylabel("monthly salary (k PLN)")
ax.legend(loc="upper left")
save(fig, "fit.png")

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(x, y, s=28, color=BLUE, edgecolor="white", linewidth=0.8)
steps = sorted(snapshots)
for i, epoch in enumerate(steps):
    w, b = snapshots[epoch]
    final = epoch == steps[-1]
    ax.plot(
        line_x,
        w * line_x + b,
        color=ORANGE,
        alpha=1.0 if final else 0.35 + 0.15 * i,
        linewidth=2 if final else 1.4,
    )
    ax.annotate(
        f"epoch {epoch}",
        xy=(10, w * 10 + b),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=MUTED,
        fontsize=9,
    )
ax.set_xlim(-0.3, 11.6)
ax.set_title("Gradient descent moves the line towards the data")
ax.set_xlabel("years of experience")
ax.set_ylabel("monthly salary (k PLN)")
save(fig, "learning.png")

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(losses, color=BLUE)
ax.set_yscale("log")
ax.set_title("The loss falls quickly, then flattens out")
ax.set_xlabel("epoch")
ax.set_ylabel("mean squared error (log scale)")
save(fig, "loss_curve.png")
