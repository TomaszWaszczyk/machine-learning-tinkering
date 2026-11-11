"""Lesson 4 - the same thing with scikit-learn, the way you would do it at work.

Run it with:

    python 04_scikit_learn.py

Everything from lesson 3 in a few library calls. The numbers differ slightly
from lesson 3 for two reasons: train_test_split picks a different split, and
scikit-learn's LogisticRegression adds a little regularisation by default
(it gently pulls the weights towards zero to reduce overfitting).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datasets import LOAN_FEATURES, make_loan_data

rows, labels = make_loan_data()
X, y = np.array(rows), np.array(labels)

# stratify=y keeps the share of defaults the same in both parts, which
# matters when one class is rare.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# A pipeline chains the scaler and the model, and guarantees that the scaler
# is fitted on the training data only.
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)

y_pred = model.predict(X_test)  # uses a threshold of 0.5
print("confusion matrix (rows: actual repaid / defaulted, columns: predicted)")
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred, target_names=["repaid", "defaulted"]))

# predict_proba gives the probabilities, so you can apply your own threshold.
p_default = model.predict_proba(X_test)[:, 1]
flagged = p_default >= 0.2
print(f"with a threshold of 0.2: {flagged.sum()} flagged, "
      f"{(flagged & (y_test == 1)).sum()} of {y_test.sum()} defaults caught\n")

applicant = [[6, 90, 30, 3]]
print("applicant: income 6k, debt 90k, age 30, 3 missed payments")
print(f"  predicted probability of default: {model.predict_proba(applicant)[0, 1]:.0%}")

scaler, classifier = model.named_steps.values()
print("\nweights in original units:")
for name, weight in zip(LOAN_FEATURES, classifier.coef_[0] / scaler.scale_):
    print(f"  {name:<17}{weight:>8.3f}")
