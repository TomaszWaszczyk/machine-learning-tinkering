"""Lesson 1 - a neural network from scratch, in pure Python.

No numpy, no scikit-learn. Run it with:

    python 01_from_scratch.py

The logistic regression tutorial ended with: "a single neuron with a sigmoid
activation IS logistic regression". This lesson takes that literally:

  Part 1  one neuron (= logistic regression) tries to learn XOR and fails
  Part 2  a network of neurons with one hidden layer learns it
  Part 3  we check our backpropagation against a brute-force calculation

The network is tiny so that every weight can be followed by hand:

    2 inputs  ->  3 hidden neurons (tanh)  ->  1 output neuron (sigmoid)

The four ingredients are the same as ever. Data, loss and optimiser are
unchanged from logistic regression; only the model is new, and with it the
way gradients are computed (backpropagation).
"""

import math
import random

from datasets import XOR_INPUTS, XOR_LABELS

HIDDEN = 3  # number of neurons in the hidden layer


def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    return math.exp(z) / (1 + math.exp(z))


def log_loss_one(p, y):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -math.log(p) if y == 1 else -math.log(1 - p)


# === Part 1: one neuron ======================================================
def train_single_neuron(inputs, labels, learning_rate=0.5, epochs=5000):
    """Logistic regression with two inputs, exactly as in the last tutorial."""
    w1, w2, b = 0.0, 0.0, 0.0
    for _ in range(epochs):
        g1 = g2 = gb = 0.0
        for (x1, x2), y in zip(inputs, labels):
            error = sigmoid(w1 * x1 + w2 * x2 + b) - y
            g1 += error * x1
            g2 += error * x2
            gb += error
        w1 -= learning_rate * g1 / len(inputs)
        w2 -= learning_rate * g2 / len(inputs)
        b -= learning_rate * gb / len(inputs)
    return [sigmoid(w1 * x1 + w2 * x2 + b) for x1, x2 in inputs]


# === Part 2: a network =======================================================
def make_network(seed=0):
    """Create the parameters, filled with small random numbers.

    W1[j][i] is the weight from input i to hidden neuron j.
    W2[j]    is the weight from hidden neuron j to the output neuron.

    The weights must NOT all start at zero: every hidden neuron would then
    compute the same thing, receive the same gradient, and stay identical
    forever. Random starting values break that symmetry.
    """
    rng = random.Random(seed)
    return {
        "W1": [[rng.uniform(-1, 1) for _ in range(2)] for _ in range(HIDDEN)],
        "b1": [0.0] * HIDDEN,
        "W2": [rng.uniform(-1, 1) for _ in range(HIDDEN)],
        "b2": 0.0,
    }


def forward(x, net):
    """The forward pass: from inputs to a prediction, layer by layer.

    Each hidden neuron does what a logistic regression does: a weighted sum
    of its inputs followed by an S-shaped function (here tanh, a sigmoid
    stretched to the range -1..1). The output neuron then does the same
    thing again, but its inputs are the hidden neurons' outputs.
    """
    h = []
    for j in range(HIDDEN):
        z = net["W1"][j][0] * x[0] + net["W1"][j][1] * x[1] + net["b1"][j]
        h.append(math.tanh(z))
    z_out = sum(net["W2"][j] * h[j] for j in range(HIDDEN)) + net["b2"]
    p = sigmoid(z_out)
    return h, p


def backward(x, y, h, p, net):
    """Backpropagation: the gradient of the loss for ONE example.

    We walk through the network backwards, and at every step ask "how much
    did this number contribute to the error?". That is the chain rule from
    calculus, applied one layer at a time.
    """
    # Output neuron. Same result as in logistic regression: prediction - truth.
    d_out = p - y
    grad_W2 = [d_out * h[j] for j in range(HIDDEN)]  # "error times input"
    grad_b2 = d_out

    grad_W1 = []
    grad_b1 = []
    for j in range(HIDDEN):
        # How much is hidden neuron j to blame? The output error, passed back
        # through the weight that connects neuron j to the output...
        d_h = d_out * net["W2"][j]
        # ...and then back through tanh, whose derivative is 1 - tanh^2.
        d_z = d_h * (1 - h[j] ** 2)
        grad_W1.append([d_z * x[0], d_z * x[1]])  # again "error times input"
        grad_b1.append(d_z)
    return {"W1": grad_W1, "b1": grad_b1, "W2": grad_W2, "b2": grad_b2}


def average_loss(inputs, labels, net):
    total = 0.0
    for x, y in zip(inputs, labels):
        _, p = forward(x, net)
        total += log_loss_one(p, y)
    return total / len(inputs)


def train(inputs, labels, net, learning_rate=0.5, epochs=5000):
    n = len(inputs)
    for epoch in range(epochs + 1):
        if epoch % 500 == 0:
            print(f"epoch {epoch:4d}   loss = {average_loss(inputs, labels, net):.4f}")

        # Add up the gradients of all examples...
        total = {"W1": [[0.0, 0.0] for _ in range(HIDDEN)], "b1": [0.0] * HIDDEN, "W2": [0.0] * HIDDEN, "b2": 0.0}
        for x, y in zip(inputs, labels):
            h, p = forward(x, net)
            g = backward(x, y, h, p, net)
            for j in range(HIDDEN):
                total["W1"][j][0] += g["W1"][j][0]
                total["W1"][j][1] += g["W1"][j][1]
                total["b1"][j] += g["b1"][j]
                total["W2"][j] += g["W2"][j]
            total["b2"] += g["b2"]

        # ...then take one gradient descent step, exactly as before.
        for j in range(HIDDEN):
            net["W1"][j][0] -= learning_rate * total["W1"][j][0] / n
            net["W1"][j][1] -= learning_rate * total["W1"][j][1] / n
            net["b1"][j] -= learning_rate * total["b1"][j] / n
            net["W2"][j] -= learning_rate * total["W2"][j] / n
        net["b2"] -= learning_rate * total["b2"] / n


# === Part 3: checking backpropagation ========================================
def gradient_check(inputs, labels, net, nudge=1e-5):
    """Compare backpropagation with the slowest, most obvious method.

    The gradient of a weight is "how much does the loss change when I nudge
    this weight a little?". So nudge it, measure the loss, and divide. This
    needs two full passes over the data PER WEIGHT, which is why nobody
    trains this way, but it is a perfect way to test that backward() is right.
    """
    n = len(inputs)
    analytic = [0.0] * HIDDEN  # we check the weights W1[j][0]
    for x, y in zip(inputs, labels):
        h, p = forward(x, net)
        g = backward(x, y, h, p, net)
        for j in range(HIDDEN):
            analytic[j] += g["W1"][j][0] / n

    worst = 0.0
    for j in range(HIDDEN):
        original = net["W1"][j][0]
        net["W1"][j][0] = original + nudge
        loss_up = average_loss(inputs, labels, net)
        net["W1"][j][0] = original - nudge
        loss_down = average_loss(inputs, labels, net)
        net["W1"][j][0] = original
        numeric = (loss_up - loss_down) / (2 * nudge)
        print(f"  W1[{j}][0]   backprop = {analytic[j]:+.8f}   nudging = {numeric:+.8f}")
        worst = max(worst, abs(numeric - analytic[j]))
    return worst


if __name__ == "__main__":
    print("XOR: the answer is 1 when exactly one input is 1")
    for x, y in zip(XOR_INPUTS, XOR_LABELS):
        print(f"  {x} -> {y}")

    print("\nPart 1: one neuron (logistic regression)")
    for x, y, p in zip(XOR_INPUTS, XOR_LABELS, train_single_neuron(XOR_INPUTS, XOR_LABELS)):
        print(f"  {x}   truth = {y}   P(1) = {p:.2f}")
    print("  It answers 0.50 for everything: no straight line separates XOR.")

    print(f"\nPart 2: a network with {HIDDEN} hidden neurons")
    net = make_network()

    print("gradient check before training (the two columns should match):")
    worst = gradient_check(XOR_INPUTS, XOR_LABELS, net)
    print(f"  largest difference: {worst:.2e}\n")

    train(XOR_INPUTS, XOR_LABELS, net)
    print()
    for x, y in zip(XOR_INPUTS, XOR_LABELS):
        h, p = forward(x, net)
        hidden = "  ".join(f"{value:+.2f}" for value in h)
        print(f"  {x}   truth = {y}   P(1) = {p:.2f}   hidden neurons: {hidden}")
    print("  The hidden layer re-describes each input with 3 new numbers, and in")
    print("  that new description a single neuron CAN separate the two classes.")
