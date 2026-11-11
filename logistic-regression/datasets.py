"""Synthetic datasets used by the lessons.

As in the linear regression tutorial, we generate the data from a known
rule so we can check whether the model recovers it. The difference is in
the answers: each example is now labelled 0 or 1 instead of with a number.

The labels are drawn at random from a probability. A student with a 70%
chance of passing really does fail 3 times out of 10, so two students with
identical inputs can have different labels. That is what real
classification data looks like.

Only the Python standard library is used here, so lesson 1 runs with no
dependencies at all.
"""

import math
import random


def _sigmoid(z):
    return 1 / (1 + math.exp(-z))


# --- Dataset 1: one feature ------------------------------------------------
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


# --- Dataset 2: two features -----------------------------------------------
# The same exam, but sleep the night before matters too.
EXAM2_FEATURES = ["hours_studied", "hours_slept"]
EXAM2_TRUE_WEIGHTS = [1.0, 0.9]
EXAM2_TRUE_BIAS = -10.5


def make_exam_data_2d(n=300, seed=3):
    """Return (rows, ys). Each row is [hours_studied, hours_slept]."""
    rng = random.Random(seed)
    rows, ys = [], []
    for _ in range(n):
        row = [rng.uniform(0, 10), rng.uniform(3, 9)]
        z = EXAM2_TRUE_BIAS + sum(w * v for w, v in zip(EXAM2_TRUE_WEIGHTS, row))
        rows.append(row)
        ys.append(1 if rng.random() < _sigmoid(z) else 0)
    return rows, ys


# --- Dataset 3: several features, rare positives ---------------------------
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
