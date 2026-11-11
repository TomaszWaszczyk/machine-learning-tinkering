"""Lesson 3 - recognising handwritten digits.

Run it with:

    python 03_digits.py

Time for real data: 1797 small greyscale images (8 x 8 pixels) of handwritten
digits, which ship with scikit-learn. The network gets the 64 pixel values
and must say which digit, 0 to 9, it is looking at.

New ideas in this lesson:

  * many classes        - 10 output neurons and the softmax function
  * ReLU                - the activation function used in modern networks
  * mini-batches        - update the weights after every 32 examples, not
                          after the whole dataset (stochastic gradient descent)
  * a validation set    - a third slice of data for watching training
  * any number of layers - the code below works for [64, 10], [64, 64, 10],
                          [64, 64, 32, 10], ... the loop is the same
"""

import numpy as np
from sklearn.datasets import load_digits

from plotting import BLUE, MUTED, ORANGE, plt, save

digits = load_digits()
X = digits.data / 16.0  # pixel values 0..16 -> 0..1, so all inputs share a scale
y = digits.target  # the digit, 0..9
images = digits.images

# --- Three-way split ----------------------------------------------------------
# train:      the network learns from this
# validation: we watch this DURING training to see how it is going
# test:       touched once, at the very end, for the honest final score
rng = np.random.default_rng(0)
order = rng.permutation(len(X))
n_test = n_val = len(X) // 5
test, val, train = order[:n_test], order[n_test : n_test + n_val], order[n_test + n_val :]
print(f"{len(train)} training, {len(val)} validation and {len(test)} test images of {X.shape[1]} pixels each\n")


def one_hot(labels, classes=10):
    """Turn the digit 3 into [0,0,0,1,0,0,0,0,0,0]: one target per output neuron."""
    out = np.zeros((len(labels), classes))
    out[np.arange(len(labels)), labels] = 1
    return out


def softmax(z):
    """Turn 10 raw scores into 10 probabilities that add up to 1.

    It is the many-class version of the sigmoid: larger scores get a larger
    share of the probability.
    """
    z = z - z.max(axis=1, keepdims=True)  # avoids overflow, does not change the result
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def make_network(sizes, seed=0):
    """sizes = [64, 64, 10] means 64 inputs, one hidden layer of 64, 10 outputs."""
    rng = np.random.default_rng(seed)
    layers = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        # "He initialisation": random weights scaled so that signals neither
        # explode nor fade as they pass through many ReLU layers.
        W = rng.normal(0, np.sqrt(2 / n_in), size=(n_in, n_out))
        layers.append([W, np.zeros(n_out)])
    return layers


def forward(layers, X):
    """Return the output of every layer; the last one holds the probabilities."""
    activations = [X]
    for i, (W, b) in enumerate(layers):
        z = activations[-1] @ W + b
        if i == len(layers) - 1:
            activations.append(softmax(z))  # output layer
        else:
            activations.append(np.maximum(z, 0))  # ReLU: keep positives, zero out negatives
    return activations


def backward(layers, activations, targets):
    """Backpropagation for any number of layers. Returns [(grad_W, grad_b), ...]."""
    grads = [None] * len(layers)
    # Softmax with cross-entropy gives the familiar starting point:
    # prediction minus truth.
    delta = (activations[-1] - targets) / len(targets)
    for i in reversed(range(len(layers))):
        grads[i] = (activations[i].T @ delta, delta.sum(axis=0))  # "error times input"
        if i > 0:
            # Pass the error back through this layer's weights, then through
            # the ReLU of the layer below (derivative: 1 where it was active).
            delta = (delta @ layers[i][0].T) * (activations[i] > 0)
    return grads


def evaluate(layers, idx):
    """Cross-entropy loss and accuracy on the given examples."""
    p = forward(layers, X[idx])[-1]
    loss = -np.mean(np.log(np.clip(p[np.arange(len(idx)), y[idx]], 1e-12, 1)))
    accuracy = np.mean(p.argmax(axis=1) == y[idx])
    return float(loss), float(accuracy)


def train_network(sizes, epochs=60, batch_size=32, learning_rate=0.3, seed=0):
    layers = make_network(sizes, seed)
    rng = np.random.default_rng(seed)
    targets = one_hot(y)
    history = []
    for epoch in range(1, epochs + 1):
        # Mini-batch gradient descent: shuffle, then update after every batch.
        # 34 small noisy steps per epoch get further than 1 big exact step.
        shuffled = rng.permutation(train)
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            activations = forward(layers, X[batch])
            grads = backward(layers, activations, targets[batch])
            for layer, (grad_W, grad_b) in zip(layers, grads):
                layer[0] -= learning_rate * grad_W
                layer[1] -= learning_rate * grad_b
        history.append((*evaluate(layers, train), *evaluate(layers, val)))
    return layers, history


def count_parameters(layers):
    return sum(W.size + b.size for W, b in layers)


# --- Baseline: no hidden layer -------------------------------------------------
# [64, 10] is "softmax regression": logistic regression for 10 classes.
baseline, _ = train_network([64, 10])
print(f"no hidden layer  [64, 10]      {count_parameters(baseline):>5} parameters   "
      f"validation accuracy = {evaluate(baseline, val)[1]:.1%}")

# --- The network ---------------------------------------------------------------
network, history = train_network([64, 64, 10])
print(f"one hidden layer [64, 64, 10]  {count_parameters(network):>5} parameters   "
      f"validation accuracy = {evaluate(network, val)[1]:.1%}\n")

print("training the [64, 64, 10] network:")
print("  epoch   train loss   train acc   val loss   val acc")
for epoch in (1, 2, 5, 10, 20, 40, 60):
    tl, ta, vl, va = history[epoch - 1]
    print(f"  {epoch:>5}   {tl:>10.4f}   {ta:>9.1%}   {vl:>8.4f}   {va:>7.1%}")

# --- The final, honest score ----------------------------------------------------
test_loss, test_accuracy = evaluate(network, test)
best_epoch = min(range(len(history)), key=lambda i: history[i][2]) + 1
print(f"  validation loss was lowest at epoch {best_epoch} ({history[best_epoch - 1][2]:.4f})")

print(f"\ntest accuracy of the network:       {test_accuracy:.1%}")
print(f"test accuracy with no hidden layer: {evaluate(baseline, test)[1]:.1%}")

p_test = forward(network, X[test])[-1]
predicted = p_test.argmax(axis=1)
wrong = np.flatnonzero(predicted != y[test])
print(f"mistakes: {len(wrong)} of {len(test)} test images")

# --- Figures ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 10, figsize=(10, 1.9))
for ax, i in zip(axes, train[:10]):
    ax.imshow(images[i], cmap="gray_r")
    ax.set_title(str(y[i]), fontsize=10, loc="center", fontweight="normal", color=MUTED)
    ax.axis("off")
fig.suptitle("The data: 8 × 8 pixel images and their labels", x=0.02, ha="left", fontweight="bold", fontsize=12)
save(fig, "digits.png")

epochs = np.arange(1, len(history) + 1)
train_loss = [h[0] for h in history]
val_loss = [h[2] for h in history]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(epochs, train_loss, color=BLUE, label="training loss")
ax.plot(epochs, val_loss, color=ORANGE, label="validation loss")
ax.set_yscale("log")
ax.set_title("Training loss keeps falling; validation loss levels off")
ax.set_xlabel("epoch")
ax.set_ylabel("cross-entropy loss (log scale)")
ax.legend(loc="upper right")
save(fig, "training_curves.png")

shown = wrong[:8]
fig, axes = plt.subplots(1, len(shown), figsize=(1.25 * len(shown), 2.1))
for ax, k in zip(np.atleast_1d(axes), shown):
    i = test[k]
    ax.imshow(images[i], cmap="gray_r")
    ax.set_title(f"is {y[i]}\nsaid {predicted[k]}", fontsize=9, loc="center", fontweight="normal", color=MUTED)
    ax.axis("off")
fig.suptitle("Test images the network got wrong", x=0.02, ha="left", fontweight="bold", fontsize=12)
save(fig, "mistakes.png")
