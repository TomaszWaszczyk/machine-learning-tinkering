"""Lesson 2 - numpy, the S-curve, and the decision boundary.

Run it with:

    python 02_numpy_and_decision_boundary.py

Three things happen here:

  * The model from lesson 1 is rewritten with numpy (vectorised).
  * We draw what it learned: an S-shaped probability curve, and next to it
    the straight line that linear regression would fit to the same 0/1
    labels, to see why a line is the wrong shape for this job.
  * We add a second feature and draw the decision boundary. Despite the
    curvy sigmoid, the boundary between the two classes is a straight line.

The script also draws the figures used in the README.
"""

import numpy as np

from datasets import EXAM2_FEATURES, make_exam_data, make_exam_data_2d
from plotting import BLUE, INK, MUTED, ORANGE, plt, save


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_loss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def train(X, y, learning_rate, epochs):
    """Gradient descent for any number of features. X has shape (n, features)."""
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = []
    for _ in range(epochs):
        p = sigmoid(X @ w + b)  # all predictions at once
        losses.append(log_loss(p, y))
        error = p - y
        w -= learning_rate * (X.T @ error) / len(y)
        b -= learning_rate * np.mean(error)
    return w, b, losses


# === Part 1: one feature =====================================================
xs, ys = make_exam_data()
x = np.array(xs)
y = np.array(ys)

w, b, losses = train(x.reshape(-1, 1), y, learning_rate=0.5, epochs=5000)
w = w[0]
print("one feature (hours studied)")
print(f"  logistic regression:  P(pass) = sigmoid({w:.3f} * hours + {b:.3f})")
print(f"  decision boundary at {-b / w:.2f} hours")

# For comparison: an ordinary least-squares line fitted to the 0/1 labels.
line_w, line_b = np.polyfit(x, y, 1)
print(f"  linear regression:    y = {line_w:.3f} * hours + {line_b:.3f}")
print(f"    its 'probability' at  0 hours: {line_b:.2f}")
print(f"    its 'probability' at 10 hours: {line_w * 10 + line_b:.2f}")

grid = np.linspace(0, 10, 200)
rng = np.random.default_rng(0)
jitter = rng.uniform(-0.03, 0.03, size=len(y))  # spread the dots so they overlap less

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(x, y + jitter, s=22, color=BLUE, alpha=0.55, edgecolor="none", label="students (1 = passed)")
ax.plot(grid, line_w * grid + line_b, color=MUTED, linewidth=1.4, linestyle="--", label="linear regression")
ax.plot(grid, sigmoid(w * grid + b), color=ORANGE, label="logistic regression")
ax.axvline(-b / w, color=INK, linewidth=0.8, linestyle=":")
ax.annotate(
    "decision boundary\nP(pass) = 0.5",
    xy=(-b / w, 0.5),
    xytext=(8, -26),
    textcoords="offset points",
    color=MUTED,
    fontsize=9,
)
ax.set_title("A line leaves the 0-1 range; the sigmoid cannot")
ax.set_xlabel("hours studied")
ax.set_ylabel("probability of passing")
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.7))
save(fig, "sigmoid_fit.png")

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(losses, color=BLUE)
ax.set_title("Log loss falls as the model trains")
ax.set_xlabel("epoch")
ax.set_ylabel("log loss")
save(fig, "loss_curve.png")

# The sigmoid itself, for the README.
z = np.linspace(-8, 8, 200)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(z, sigmoid(z), color=ORANGE)
ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
ax.set_title("The sigmoid turns any number into a probability")
ax.set_xlabel("z = w·x + b")
ax.set_ylabel("sigmoid(z)")
save(fig, "sigmoid.png")

# === Part 2: two features ====================================================
rows, labels = make_exam_data_2d()
X2 = np.array(rows)
y2 = np.array(labels)

# Standardise the features so gradient descent converges quickly (see the
# feature-scaling section of the linear regression tutorial).
mean, std = X2.mean(axis=0), X2.std(axis=0)
w_s, b_s, _ = train((X2 - mean) / std, y2, learning_rate=0.5, epochs=3000)

# Back to the original units so the weights can be read.
w2 = w_s / std
b2 = b_s - np.sum(w_s * mean / std)

print("\ntwo features")
for name, weight in zip(EXAM2_FEATURES, w2):
    print(f"  {name:<14}{weight:>7.3f}")
print(f"  {'bias':<14}{b2:>7.3f}")
accuracy = np.mean((sigmoid(X2 @ w2 + b2) >= 0.5) == (y2 == 1))
print(f"  training accuracy: {accuracy:.1%}")

# The boundary is where w1*x1 + w2*x2 + b = 0. Solve for x2: a straight line.
study = np.array([0, 10])
sleep_on_boundary = -(w2[0] * study + b2) / w2[1]

fig, ax = plt.subplots(figsize=(8, 4.4))
passed = y2 == 1
ax.scatter(X2[~passed, 0], X2[~passed, 1], s=30, color=ORANGE, marker="x", linewidth=1.4, label="failed")
ax.scatter(X2[passed, 0], X2[passed, 1], s=28, color=BLUE, edgecolor="white", linewidth=0.8, label="passed")
ax.plot(study, sleep_on_boundary, color=INK, linewidth=1.6, label="decision boundary")
ax.set_xlim(-0.3, 10.3)
ax.set_ylim(2.8, 9.2)
ax.set_title("With two features the decision boundary is a straight line")
ax.set_xlabel("hours studied")
ax.set_ylabel("hours slept")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
save(fig, "decision_boundary.png")
