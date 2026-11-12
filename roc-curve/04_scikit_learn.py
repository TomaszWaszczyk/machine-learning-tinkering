"""Lesson 4 - the same thing with scikit-learn, the way you would do it at work.

Run it with:

    python 04_scikit_learn.py

Everything from lessons 1-3 in a few library calls. The numbers differ
slightly from lesson 2 because train_test_split picks a different split and
LogisticRegression adds a little regularisation by default.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datasets import make_loan_data

rows, labels = make_loan_data()
X, y = np.array(rows), np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)

# ROC needs SCORES, not yes/no answers. Pass predict_proba, never predict:
# with 0/1 predictions the "curve" has a single point and the AUC is wrong.
p_default = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, p_default)
print(f"roc_curve returned {len(fpr)} points")
print(f"AUC               = {roc_auc_score(y_test, p_default):.3f}")
print(f"average precision = {average_precision_score(y_test, p_default):.3f}")
print(f"AUC from predict() instead of predict_proba(): "
      f"{roc_auc_score(y_test, model.predict(X_test)):.3f}  <- the classic mistake\n")

# Choosing a threshold from the curve: the lowest false positive rate that
# still catches at least 80% of the defaults.
i = int(np.argmax(tpr >= 0.8))
print("to catch at least 80% of defaults:")
print(f"  threshold = {thresholds[i]:.3f}   TPR = {tpr[i]:.1%}   FPR = {fpr[i]:.1%}\n")

# One test set of 400 borrowers has only about 60 defaults, so its AUC is a
# noisy estimate. Cross-validation gives five estimates instead of one.
print("5-fold cross-validated AUC")
candidates = {
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression()),
    "random forest": RandomForestClassifier(n_estimators=200, random_state=0),
}
for name, candidate in candidates.items():
    aucs = cross_val_score(candidate, X, y, cv=5, scoring="roc_auc")
    print(f"  {name:<20} {aucs.mean():.3f} +/- {aucs.std():.3f}   folds: {np.round(aucs, 3)}")

# To draw the curve, scikit-learn has a one-liner:
#
#     from sklearn.metrics import RocCurveDisplay
#     RocCurveDisplay.from_estimator(model, X_test, y_test)
