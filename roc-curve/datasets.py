"""Synthetic datasets used by the lessons.

These are the exam and loan datasets from the logistic regression tutorial,
plus a tiny hand-made one that is small enough to work through on paper.

Only the Python standard library is used here, so lesson 1 runs with no
dependencies at all.
"""

import math
import random


def _sigmoid(z):
    return 1 / (1 + math.exp(-z))


# --- Dataset 1: ten borrowers, small enough to do by hand -------------------
# A model gave each borrower a score: its estimate of the probability of
# default. The label says what really happened (1 = defaulted, 0 = repaid).
# Listed from the highest score to the lowest.
TOY_SCORES = [0.95, 0.85, 0.80, 0.70, 0.55, 0.45, 0.40, 0.30, 0.20, 0.10]
TOY_LABELS = [1, 1, 0, 1, 0, 1, 0, 0, 0, 0]


# --- Dataset 2: one feature --------------------------------------------------
# Did the student pass the exam (1) or fail (0), given hours studied?
EXAM_TRUE_WEIGHT = 1.5
EXAM_TRUE_BIAS = -7.5  # so the 50/50 point is at 7.5 / 1.5 = 5 hours


def make_exam_data(n=200, seed=42):
    """Return (xs, ys): hours studied and 1 = passed / 0 = failed."""
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        hours = rng.uniform(0, 10)
        p_pass = _sigmoid(EXAM_TRUE_WEIGHT * hours + EXAM_TRUE_BIAS)
        xs.append(hours)
        ys.append(1 if rng.random() < p_pass else 0)
    return xs, ys


# --- Dataset 3: several features, rare positives ----------------------------
# Will a borrower default on a loan (1) or repay it (0)?
LOAN_FEATURES = ["income_k_pln", "debt_k_pln", "age_years", "missed_payments"]
LOAN_TRUE_WEIGHTS = [-0.18, 0.02, -0.02, 0.9]
LOAN_TRUE_BIAS = -2.2


def make_loan_data(n=2000, seed=11):
    """Return (rows, ys). Most borrowers repay: defaults are the rare class."""
    rng = random.Random(seed)
    rows, ys = [], []
    for _ in range(n):
        income = rng.uniform(3, 20)  # monthly, thousand PLN
        debt = rng.uniform(0, 150)  # existing debt, thousand PLN
        age = rng.uniform(21, 65)
        missed = rng.choices([0, 1, 2, 3, 4, 5], weights=[50, 22, 13, 8, 4, 3])[0]
        row = [income, debt, age, missed]
        z = LOAN_TRUE_BIAS + sum(w * v for w, v in zip(LOAN_TRUE_WEIGHTS, row))
        rows.append(row)
        ys.append(1 if rng.random() < _sigmoid(z) else 0)
    return rows, ys
