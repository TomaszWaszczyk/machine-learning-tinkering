"""The loan-default classifier from the logistic regression tutorial.

This tutorial is about judging a classifier, not about training one, so the
training code lives here, out of the way. It is the same logistic regression
as in ../logistic-regression/03_evaluating_a_classifier.py, with the same
train/test split, so the numbers match that lesson exactly.
"""

import numpy as np

from datasets import LOAN_FEATURES, make_loan_data


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loan_test_scores(features=LOAN_FEATURES, n=2000):
    """Train on 80% of the loan data and score the remaining 20%.

    Returns (y_test, p_test): the true labels of the test set (1 = defaulted)
    and the model's predicted probability of default for each borrower.
    `features` chooses which columns the model is allowed to see.
    """
    rows, labels = make_loan_data(n=n)
    columns = [LOAN_FEATURES.index(name) for name in features]
    X = np.array(rows)[:, columns]
    y = np.array(labels)

    order = np.random.default_rng(0).permutation(len(X))
    n_test = len(X) // 5
    test, train = order[:n_test], order[n_test:]

    mean, std = X[train].mean(axis=0), X[train].std(axis=0)
    X_train = (X[train] - mean) / std
    X_test = (X[test] - mean) / std

    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(3000):
        error = sigmoid(X_train @ w + b) - y[train]
        w -= 0.5 * (X_train.T @ error) / len(train)
        b -= 0.5 * np.mean(error)

    return y[test], sigmoid(X_test @ w + b)
