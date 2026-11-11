"""Lesson 3 - is the classifier any good? Accuracy is not enough.

Run it with:

    python 03_evaluating_a_classifier.py

We predict whether a borrower will default on a loan, from four features.
Defaults are rare, and that changes how a model must be judged.

New ideas in this lesson:

  * the baseline          - what score does a model that learned nothing get?
  * the confusion matrix  - the four kinds of right and wrong answers
  * precision and recall  - two questions accuracy cannot answer
  * the threshold         - 0.5 is a choice, not a law
  * odds ratios           - reading the weights of a logistic model
"""

import numpy as np

from datasets import LOAN_FEATURES, LOAN_TRUE_WEIGHTS, make_loan_data
from plotting import BLUE, MUTED, ORANGE, plt, save

rows, labels = make_loan_data()
X = np.array(rows)
y = np.array(labels)

# --- Train / test split and scaling (as in the linear regression tutorial) ---
rng = np.random.default_rng(0)
order = rng.permutation(len(X))
n_test = len(X) // 5
test, train = order[:n_test], order[n_test:]
X_train, y_train = X[train], y[train]
X_test, y_test = X[test], y[test]

mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train_s = (X_train - mean) / std
X_test_s = (X_test - mean) / std  # training mean and std, never the test set's

print(f"{len(X_train)} training examples, {len(X_test)} test examples")
print(f"share of defaults in the test set: {y_test.mean():.1%}\n")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_logistic(X, y, learning_rate=0.5, epochs=3000):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        error = sigmoid(X @ w + b) - y
        w -= learning_rate * (X.T @ error) / len(y)
        b -= learning_rate * np.mean(error)
    return w, b


w_s, b_s = train_logistic(X_train_s, y_train)
p_test = sigmoid(X_test_s @ w_s + b_s)  # predicted probability of default


def confusion(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))  # true positive: caught a default
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))  # false positive: false alarm
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))  # false negative: missed a default
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))  # true negative: correctly cleared
    return tp, fp, fn, tn


def scores(y_true, y_pred):
    tp, fp, fn, tn = confusion(y_true, y_pred)
    accuracy = (tp + tn) / len(y_true)
    # Precision: of the people we flagged, how many really defaulted?
    precision = tp / (tp + fp) if tp + fp else 0.0
    # Recall: of the people who really defaulted, how many did we flag?
    recall = tp / (tp + fn) if tp + fn else 0.0
    return accuracy, precision, recall


# --- The baseline -----------------------------------------------------------
# A "model" that always answers "will repay" is right most of the time,
# simply because most people repay. Any real model must beat this number.
always_repay = np.zeros_like(y_test)
acc, prec, rec = scores(y_test, always_repay)
print("baseline: always predict 'will repay'")
print(f"  accuracy = {acc:.1%}   recall = {rec:.1%}   <- catches no defaults at all\n")

# --- Our model at the default threshold of 0.5 -------------------------------
y_pred = (p_test >= 0.5).astype(int)
tp, fp, fn, tn = confusion(y_test, y_pred)
acc, prec, rec = scores(y_test, y_pred)
print("logistic regression, threshold 0.5")
print("                        predicted default   predicted repay")
print(f"  actually defaulted    {tp:>10}          {fn:>10}")
print(f"  actually repaid       {fp:>10}          {tn:>10}")
print(f"  accuracy = {acc:.1%}   precision = {prec:.1%}   recall = {rec:.1%}\n")

# --- Moving the threshold ----------------------------------------------------
# The model outputs a probability. Where we cut it into yes/no is OUR choice,
# and it should depend on what each kind of mistake costs. Missing a default
# is expensive for a bank, so it may prefer to flag anyone above 20%.
print("the same model at different thresholds")
print("  threshold   flagged   precision   recall   accuracy")
for threshold in (0.8, 0.5, 0.3, 0.2, 0.1):
    y_pred = (p_test >= threshold).astype(int)
    acc, prec, rec = scores(y_test, y_pred)
    print(f"  {threshold:>7.1f}   {y_pred.sum():>7}   {prec:>8.1%}   {rec:>6.1%}   {acc:>7.1%}")

# --- Reading the weights -----------------------------------------------------
# In original units each weight is the change in the LOG-ODDS of default per
# unit of the feature. exp(weight) is easier to say out loud: the factor by
# which the odds are multiplied.
w = w_s / std
print("\nwhat the model learned (original units)")
print(f"  {'feature':<17}{'learned':>9}{'true':>8}{'odds x':>9}")
for name, learned, true in zip(LOAN_FEATURES, w, LOAN_TRUE_WEIGHTS):
    print(f"  {name:<17}{learned:>9.3f}{true:>8.3f}{np.exp(learned):>9.2f}")

# --- Figure ------------------------------------------------------------------
thresholds = np.linspace(0.02, 0.9, 89)
precisions, recalls = [], []
for threshold in thresholds:
    _, prec, rec = scores(y_test, (p_test >= threshold).astype(int))
    # Precision is undefined when nothing is flagged; leave a gap in the line.
    precisions.append(prec if (p_test >= threshold).any() else np.nan)
    recalls.append(rec)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(thresholds, precisions, color=BLUE, label="precision")
ax.plot(thresholds, recalls, color=ORANGE, label="recall")
ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
ax.set_ylim(0, 1.02)
ax.set_title("Raising the threshold trades recall for precision")
ax.set_xlabel("threshold on the predicted probability of default")
ax.set_ylabel("score on the test set")
ax.legend(loc="center right")
save(fig, "precision_recall.png")
