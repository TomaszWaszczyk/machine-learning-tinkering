"""Small datasets used by lessons 1 and 2.

Both are chosen because a single straight line cannot separate the classes,
which is exactly the situation where logistic regression fails and a neural
network is needed.

Only the Python standard library is used here, so lesson 1 runs with no
dependencies at all. Lesson 3 uses real data (handwritten digits) that ships
with scikit-learn.
"""

import math
import random

# --- Dataset 1: XOR ----------------------------------------------------------
# "Exclusive or": the answer is 1 when exactly one of the two inputs is 1.
# The smallest problem that no straight line can solve.
XOR_INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_LABELS = [0, 1, 1, 0]


# --- Dataset 2: a disc inside a ring ------------------------------------------
def make_ring_data(n=400, seed=1):
    """Return (rows, ys). Class 1 is a disc at the centre, class 0 a ring around it."""
    rng = random.Random(seed)
    rows, ys = [], []
    for i in range(n):
        label = i % 2
        if label == 1:
            radius = 1.0 * math.sqrt(rng.random())  # spread evenly over the disc
        else:
            radius = rng.uniform(1.6, 2.6)
        angle = rng.uniform(0, 2 * math.pi)
        x1 = radius * math.cos(angle) + rng.gauss(0, 0.12)
        x2 = radius * math.sin(angle) + rng.gauss(0, 0.12)
        rows.append([x1, x2])
        ys.append(label)
    return rows, ys
