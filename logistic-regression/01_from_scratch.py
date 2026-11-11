"""Lesson 1 - logistic regression from scratch, in pure Python.

No numpy, no scikit-learn. Run it with:

    python 01_from_scratch.py

Compare this file with 01_from_scratch.py in the linear regression tutorial.
The four ingredients are the same; only two of them change:

    1. data        - the answers are now 0 or 1 instead of numbers
    2. model       - the straight line is squashed into a probability   <- new
    3. loss        - log loss instead of mean squared error             <- new
    4. optimiser   - gradient descent, exactly as before
"""

import math

from datasets import EXAM_TRUE_BIAS, EXAM_TRUE_WEIGHT, make_exam_data


# 2. MODEL -------------------------------------------------------------------
def sigmoid(z):
    """Squash any number into the range (0, 1).

    Large positive z -> close to 1, large negative z -> close to 0, z = 0 -> 0.5.
    The two branches compute the same thing; the second avoids an overflow
    in math.exp for very negative z.
    """
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    return math.exp(z) / (1 + math.exp(z))


def predict_probability(x, w, b):
    """The whole model: a straight line fed through the sigmoid.

    The result is read as "the probability that the answer is 1".
    """
    return sigmoid(w * x + b)


# 3. LOSS --------------------------------------------------------------------
def log_loss(xs, ys, w, b):
    """Average "surprise" of the model at the true answers.

    For one example with predicted probability p:
        if the truth is 1 the loss is  -log(p)
        if the truth is 0 the loss is  -log(1 - p)

    Being confident and right costs almost nothing. Being confident and
    wrong costs a lot: -log(0.01) is about 4.6, and it grows without limit.
    """
    total = 0.0
    for x, y in zip(xs, ys):
        p = predict_probability(x, w, b)
        p = min(max(p, 1e-12), 1 - 1e-12)  # never take log(0)
        total += -math.log(p) if y == 1 else -math.log(1 - p)
    return total / len(xs)


# 4. OPTIMISER ---------------------------------------------------------------
def gradients(xs, ys, w, b):
    """Partial derivatives of the log loss.

        dL/dw = (1/n) * sum( (p - y) * x )
        dL/db = (1/n) * sum( (p - y) )

    Look familiar? It is "error times input", the same shape as in linear
    regression. The only difference is that the prediction p now comes out
    of the sigmoid.
    """
    n = len(xs)
    grad_w = 0.0
    grad_b = 0.0
    for x, y in zip(xs, ys):
        error = predict_probability(x, w, b) - y
        grad_w += error * x
        grad_b += error
    return grad_w / n, grad_b / n


def train(xs, ys, learning_rate=0.5, epochs=5000):
    """Gradient descent: identical loop to the linear regression tutorial."""
    w, b = 0.0, 0.0
    for epoch in range(epochs + 1):
        if epoch % 500 == 0:
            loss = log_loss(xs, ys, w, b)
            print(f"epoch {epoch:4d}   w = {w:6.3f}   b = {b:7.3f}   loss = {loss:.4f}")
        grad_w, grad_b = gradients(xs, ys, w, b)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
    return w, b


if __name__ == "__main__":
    # 1. DATA
    xs, ys = make_exam_data()
    print(f"{len(xs)} students, {sum(ys)} passed. The first five:")
    for x, y in list(zip(xs, ys))[:5]:
        print(f"  studied {x:4.1f} h -> {'passed' if y else 'failed'}")
    print()

    w, b = train(xs, ys)

    print()
    print(f"learned:  P(pass) = sigmoid({w:.3f} * hours + {b:.3f})")
    print(f"truth:    P(pass) = sigmoid({EXAM_TRUE_WEIGHT:.3f} * hours + {EXAM_TRUE_BIAS:.3f})")
    print()
    for hours in (2, 4, 5, 6, 8):
        p = predict_probability(hours, w, b)
        verdict = "pass" if p >= 0.5 else "fail"
        print(f"  {hours} hours of study: P(pass) = {p:.2f}  -> predict {verdict}")
    print()
    # The model says 50/50 exactly where w*x + b = 0.
    print(f"decision boundary: {-b / w:.2f} hours (truth: {-EXAM_TRUE_BIAS / EXAM_TRUE_WEIGHT:.2f})")

    correct = sum((predict_probability(x, w, b) >= 0.5) == (y == 1) for x, y in zip(xs, ys))
    print(f"accuracy on the training data: {correct}/{len(xs)} = {correct / len(xs):.1%}")
