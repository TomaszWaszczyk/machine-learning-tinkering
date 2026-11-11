"""Lesson 2 - the same network with numpy, and what the hidden layer buys us.

Run it with:

    python 02_numpy_network.py

Lesson 1 looped over neurons one by one. Here a whole layer is one matrix
multiplication, so the network is a handful of lines:

    H = tanh(X @ W1 + b1)        hidden layer
    P = sigmoid(H @ W2 + b2)     output layer

The data is a disc of one class surrounded by a ring of the other. We train:

  * logistic regression (no hidden layer)   -> a straight boundary, useless here
  * networks with 1, 2, 3 and 8 hidden neurons, to see the boundary take shape

The script also draws the figures used in the README.
"""

import numpy as np

from datasets import make_ring_data
from plotting import BLUE, INK, ORANGE, plt, save

rows, labels = make_ring_data()
X = np.array(rows)  # shape (400, 2)
y = np.array(labels).reshape(-1, 1)  # shape (400, 1): a column, like the network's output


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_loss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# --- Logistic regression, for comparison -------------------------------------
def train_logistic(X, y, learning_rate=0.5, epochs=3000):
    w = np.zeros((X.shape[1], 1))
    b = 0.0
    for _ in range(epochs):
        error = sigmoid(X @ w + b) - y
        w -= learning_rate * X.T @ error / len(y)
        b -= learning_rate * error.mean()
    return lambda points: sigmoid(points @ w + b)


# --- The network --------------------------------------------------------------
def train_network(X, y, hidden, learning_rate=0.5, epochs=5000, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1, size=(2, hidden))  # inputs -> hidden
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1, size=(hidden, 1))  # hidden -> output
    b2 = 0.0
    n = len(y)
    losses = []

    for _ in range(epochs):
        # Forward pass: all 400 examples through both layers at once.
        H = np.tanh(X @ W1 + b1)  # shape (400, hidden)
        P = sigmoid(H @ W2 + b2)  # shape (400, 1)
        losses.append(log_loss(P, y))

        # Backward pass: the same four steps as in lesson 1, as matrices.
        d_out = (P - y) / n  # error at the output
        grad_W2 = H.T @ d_out  # "error times input" for the output layer
        grad_b2 = d_out.sum()
        d_hidden = (d_out @ W2.T) * (1 - H**2)  # error passed back through W2 and tanh
        grad_W1 = X.T @ d_hidden  # "error times input" for the hidden layer
        grad_b1 = d_hidden.sum(axis=0)

        W1 -= learning_rate * grad_W1
        b1 -= learning_rate * grad_b1
        W2 -= learning_rate * grad_W2
        b2 -= learning_rate * grad_b2

    def predict(points):
        return sigmoid(np.tanh(points @ W1 + b1) @ W2 + b2)

    return predict, losses


def accuracy(predict):
    return float(np.mean((predict(X) >= 0.5) == (y == 1)))


logistic = train_logistic(X, y)
print(f"logistic regression (no hidden layer):  accuracy = {accuracy(logistic):.1%}")

networks = {}
for hidden in (1, 2, 3, 8):
    predict, losses = train_network(X, y, hidden)
    networks[hidden] = (predict, losses)
    print(f"network with {hidden} hidden neuron(s):         accuracy = {accuracy(predict):.1%}   final loss = {losses[-1]:.4f}")

n_weights = 2 * 8 + 8 + 8 + 1
print(f"\nthe 8-neuron network has {n_weights} parameters: 16 + 8 in the hidden layer, 8 + 1 in the output")


# --- Figures ------------------------------------------------------------------
def draw_decision_regions(ax, predict, title):
    """Shade the area where the model predicts each class, then draw the data."""
    grid_x, grid_y = np.meshgrid(np.linspace(-3.2, 3.2, 300), np.linspace(-3.2, 3.2, 300))
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    p = predict(points).reshape(grid_x.shape)
    ax.contourf(grid_x, grid_y, p, levels=[0, 0.5, 1], colors=[ORANGE, BLUE], alpha=0.14)
    ax.contour(grid_x, grid_y, p, levels=[0.5], colors=[INK], linewidths=1.4)
    inner = y.ravel() == 1
    ax.scatter(X[~inner, 0], X[~inner, 1], s=14, color=ORANGE, marker="x", linewidth=1.1, label="class 0 (ring)")
    ax.scatter(X[inner, 0], X[inner, 1], s=13, color=BLUE, edgecolor="white", linewidth=0.4, label="class 1 (disc)")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


fig, axes = plt.subplots(1, 2, figsize=(8, 4.3))
draw_decision_regions(axes[0], logistic, f"Logistic regression: {accuracy(logistic):.1%} correct")
draw_decision_regions(axes[1], networks[8][0], f"8 hidden neurons: {accuracy(networks[8][0]):.1%} correct")
axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.02), ncol=2)
fig.suptitle("A hidden layer lets the boundary bend", x=0.02, ha="left", fontweight="bold", fontsize=12)
save(fig, "line_vs_network.png")

fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
for ax, hidden in zip(axes, networks):
    predict = networks[hidden][0]
    draw_decision_regions(ax, predict, f"{hidden} hidden: {accuracy(predict):.1%} correct")
fig.suptitle("Each hidden neuron adds one straight cut", x=0.02, ha="left", fontweight="bold", fontsize=12)
save(fig, "hidden_neurons.png")

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(networks[8][1], color=BLUE)
ax.set_yscale("log")
ax.set_title("Training loss of the 8-neuron network")
ax.set_xlabel("epoch")
ax.set_ylabel("log loss (log scale)")
save(fig, "loss_curve.png")
