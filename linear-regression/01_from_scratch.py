"""Lesson 1 - linear regression from scratch, in pure Python.

No numpy, no scikit-learn: every number is computed by code you can read
below. Run it with:

    python 01_from_scratch.py

The four ingredients of (almost) every machine-learning algorithm appear
here in their simplest form:

    1. data        - examples of inputs x with known answers y
    2. model       - a formula with adjustable numbers (parameters)
    3. loss        - one number that says how wrong the model currently is
    4. optimiser   - a procedure that changes the parameters to lower the loss
"""

from datasets import SALARY_TRUE_BIAS, SALARY_TRUE_WEIGHT, make_salary_data


# 2. MODEL -------------------------------------------------------------------
def predict(x, w, b):
    """The whole model: a straight line. w is the slope, b the intercept."""
    return w * x + b


# 3. LOSS --------------------------------------------------------------------
def mean_squared_error(xs, ys, w, b):
    """Average of the squared differences between prediction and truth."""
    total = 0.0
    for x, y in zip(xs, ys):
        error = predict(x, w, b) - y
        total += error**2
    return total / len(xs)


# 4. OPTIMISER ---------------------------------------------------------------
def gradients(xs, ys, w, b):
    """How does the loss change if we nudge w or b a little?

    These two formulas are the partial derivatives of the mean squared error:

        dL/dw = (2/n) * sum( (w*x + b - y) * x )
        dL/db = (2/n) * sum( (w*x + b - y) )

    A positive gradient means "increasing this parameter increases the loss",
    so we should decrease it - and the other way round.
    """
    n = len(xs)
    grad_w = 0.0
    grad_b = 0.0
    for x, y in zip(xs, ys):
        error = predict(x, w, b) - y
        grad_w += error * x
        grad_b += error
    return 2 * grad_w / n, 2 * grad_b / n


def train(xs, ys, learning_rate=0.01, epochs=2000):
    """Gradient descent: repeatedly take a small step downhill."""
    w, b = 0.0, 0.0  # start with a model that knows nothing
    history = []
    for epoch in range(epochs + 1):
        loss = mean_squared_error(xs, ys, w, b)
        history.append(loss)
        if epoch % 200 == 0:
            print(f"epoch {epoch:4d}   w = {w:6.3f}   b = {b:6.3f}   loss = {loss:8.4f}")

        grad_w, grad_b = gradients(xs, ys, w, b)
        w -= learning_rate * grad_w  # step AGAINST the gradient
        b -= learning_rate * grad_b
    return w, b, history


if __name__ == "__main__":
    # 1. DATA
    xs, ys = make_salary_data()
    print(f"{len(xs)} examples, the first three:")
    for x, y in list(zip(xs, ys))[:3]:
        print(f"  {x:4.1f} years of experience -> {y:5.2f}k PLN")
    print()

    w, b, _ = train(xs, ys)

    print()
    print(f"learned:  salary = {w:.3f} * years + {b:.3f}")
    print(f"truth:    salary = {SALARY_TRUE_WEIGHT:.3f} * years + {SALARY_TRUE_BIAS:.3f}  (+ noise)")
    print()
    print(f"prediction for 6 years of experience: {predict(6, w, b):.2f}k PLN")
