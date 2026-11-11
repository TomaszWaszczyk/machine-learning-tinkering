"""Lesson 4 - the same thing with scikit-learn, the way you would do it at work.

Run it with:

    python 04_scikit_learn.py

Everything from lesson 3 collapses into a few library calls. Nothing new
happens under the hood - which is the point: you now know what these calls do.

The numbers differ slightly from lesson 3 because train_test_split shuffles
the data into a different train/test split.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from datasets import APARTMENT_FEATURES, make_apartment_data

rows, prices = make_apartment_data()
X, y = np.array(rows), np.array(prices)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# LinearRegression solves the least-squares problem directly (like the normal
# equation in lesson 2), so it needs neither a learning rate nor scaling.
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"test RMSE = {root_mean_squared_error(y_test, y_pred):.1f}k PLN")
print(f"test R^2  = {r2_score(y_test, y_pred):.4f}\n")

for name, weight in zip(APARTMENT_FEATURES, model.coef_):
    print(f"  {name:<14}{weight:>8.2f}")
print(f"  {'bias':<14}{model.intercept_:>8.2f}")

flat = [[60, 3, 10, 4]]
print(f"\na 60 m2, 3-room, 10-year-old flat 4 km from the centre: {model.predict(flat)[0]:.0f}k PLN")
