"""Synthetic datasets used by the lessons.

We generate the data ourselves instead of downloading it. That gives us one
big advantage while learning: we KNOW the true relationship hidden in the
data, so we can check whether the model manages to recover it.

Only the Python standard library is used here, so lesson 1 runs with no
dependencies at all.
"""

import random

# --- Dataset 1: one feature ------------------------------------------------
# Monthly salary (thousand PLN) as a function of years of experience.
SALARY_TRUE_WEIGHT = 1.5  # every extra year adds 1.5k PLN
SALARY_TRUE_BIAS = 4.0  # someone with 0 years of experience earns 4k PLN


def make_salary_data(n=50, seed=42):
    """Return (xs, ys): years of experience and monthly salary in k PLN."""
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        years = rng.uniform(0, 10)
        noise = rng.gauss(0, 1.0)  # real life never sits exactly on a line
        salary = SALARY_TRUE_WEIGHT * years + SALARY_TRUE_BIAS + noise
        xs.append(years)
        ys.append(salary)
    return xs, ys


# --- Dataset 2: several features -------------------------------------------
# Apartment price (thousand PLN) as a function of four features.
APARTMENT_FEATURES = ["size_m2", "rooms", "age_years", "km_to_centre"]
APARTMENT_TRUE_WEIGHTS = [9.0, 15.0, -1.2, -12.0]
APARTMENT_TRUE_BIAS = 120.0


def make_apartment_data(n=300, seed=7):
    """Return (rows, prices). Each row is [size_m2, rooms, age, km_to_centre]."""
    rng = random.Random(seed)
    rows, prices = [], []
    for _ in range(n):
        size = rng.uniform(25, 120)
        # Bigger flats tend to have more rooms: the features are correlated,
        # just like in real data.
        rooms = max(1, round(size / 25 + rng.gauss(0, 0.6)))
        age = rng.uniform(0, 60)
        km = rng.uniform(0.5, 15)
        row = [size, rooms, age, km]
        price = APARTMENT_TRUE_BIAS + rng.gauss(0, 25)
        for weight, value in zip(APARTMENT_TRUE_WEIGHTS, row):
            price += weight * value
        rows.append(row)
        prices.append(price)
    return rows, prices
