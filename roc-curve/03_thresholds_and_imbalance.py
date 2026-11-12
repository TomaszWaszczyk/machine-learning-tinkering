"""Lesson 3 - using the curve: picking a threshold, and where ROC misleads.

Run it with:

    python 03_thresholds_and_imbalance.py

The ROC curve does not tell you which threshold to use. It lays out the
options. This lesson shows two ways to choose, and then the main trap: on
data with very few positives, a good-looking ROC curve can hide a model
whose alarms are almost all false.

New ideas in this lesson:

  * Youden's J              - the point furthest above the diagonal
  * cost-based choice       - the threshold that loses the least money
  * prevalence              - the share of positives in the data
  * precision-recall curve  - the picture to draw when positives are rare

It saves one figure into images/.
"""

import numpy as np

from loan_model import loan_test_scores
from plotting import BLUE, MUTED, ORANGE, plt, save


def roc_curve(y, scores):
    """Return (fpr, tpr, thresholds). Explained in lesson 2."""
    order = np.argsort(-scores, kind="stable")
    y_sorted, s_sorted = y[order], scores[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    keep = np.append(np.diff(s_sorted) != 0, True)
    tpr = np.concatenate([[0.0], tp[keep] / tp[-1]])
    fpr = np.concatenate([[0.0], fp[keep] / fp[-1]])
    thresholds = np.concatenate([[np.inf], s_sorted[keep]])
    return fpr, tpr, thresholds


def auc(fpr, tpr):
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


# A bigger sample than in lesson 2, so the curves are smooth and the rare
# class still has enough examples after we thin it out below.
y_test, p_test = loan_test_scores(n=20000)
n_pos, n_neg = int(y_test.sum()), int((y_test == 0).sum())
print(f"{len(y_test)} test borrowers, {n_pos} defaulted ({y_test.mean():.1%})")

fpr, tpr, thresholds = roc_curve(y_test, p_test)
print(f"AUC = {auc(fpr, tpr):.3f}\n")

# --- Choice 1: Youden's J ------------------------------------------------------
# J = TPR - FPR is the height of the curve above the diagonal. Maximising it
# treats a missed positive and a false alarm as equally bad, per case.
j = tpr - fpr
best = int(np.argmax(j))
print("Youden's J: the point furthest above the diagonal")
print(f"  threshold = {thresholds[best]:.3f}   TPR = {tpr[best]:.1%}   FPR = {fpr[best]:.1%}   J = {j[best]:.3f}\n")

# --- Choice 2: what do the mistakes cost? ---------------------------------------
# A missed default loses the bank the loan. A wrongly rejected good customer
# loses it only the profit on that loan.
COST_MISSED_DEFAULT = 50  # thousand PLN, a false negative
COST_REJECTED_REPAYER = 5  # thousand PLN, a false positive

false_negatives = (1 - tpr) * n_pos
false_positives = fpr * n_neg
cost = COST_MISSED_DEFAULT * false_negatives + COST_REJECTED_REPAYER * false_positives
cheapest = int(np.argmin(cost))

print(f"cost: missed default = {COST_MISSED_DEFAULT}k PLN, rejected repayer = {COST_REJECTED_REPAYER}k PLN")
print("  threshold      TPR     FPR   total cost (k PLN)")
for threshold in (0.5, 0.3, 0.2, 0.1, 0.05):
    i = int(np.searchsorted(-thresholds, -threshold, side="right")) - 1
    print(f"  {threshold:>7.2f}   {tpr[i]:>6.1%}  {fpr[i]:>6.1%}   {cost[i]:>12,.0f}")
print(f"  cheapest: threshold = {thresholds[cheapest]:.3f}   TPR = {tpr[cheapest]:.1%}   "
      f"FPR = {fpr[cheapest]:.1%}   cost = {cost[cheapest]:,.0f}\n")


# --- The trap: rare positives -----------------------------------------------------
def precision_recall_curve(y, scores):
    """Return (recall, precision), one point per distinct score."""
    order = np.argsort(-scores, kind="stable")
    y_sorted, s_sorted = y[order], scores[order]
    tp = np.cumsum(y_sorted)
    flagged = np.arange(1, len(y) + 1)
    keep = np.append(np.diff(s_sorted) != 0, True)
    return tp[keep] / tp[-1], tp[keep] / flagged[keep]


def average_precision(recall, precision):
    """Area under the precision-recall curve (as a step function)."""
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


# Same model, same kind of borrowers, but imagine a portfolio where defaults
# are ten times rarer: keep every repayer and only one defaulter in ten.
rng = np.random.default_rng(0)
keep_row = (y_test == 0) | (rng.random(len(y_test)) < 0.1)
y_rare, p_rare = y_test[keep_row], p_test[keep_row]

fpr_rare, tpr_rare, _ = roc_curve(y_rare, p_rare)
recall, precision = precision_recall_curve(y_test, p_test)
recall_rare, precision_rare = precision_recall_curve(y_rare, p_rare)


def at_threshold(y, p, threshold):
    flagged = p >= threshold
    tp = (flagged & (y == 1)).sum()
    return tp / (y == 1).sum(), (flagged & (y == 0)).sum() / (y == 0).sum(), tp / flagged.sum()


print("the same model on two portfolios, threshold 0.2")
print("  share of defaults     AUC     TPR     FPR   precision   avg precision")
for name_y, name_p, curve, pr in (
    (y_test, p_test, (fpr, tpr), (recall, precision)),
    (y_rare, p_rare, (fpr_rare, tpr_rare), (recall_rare, precision_rare)),
):
    t, f, prec = at_threshold(name_y, name_p, 0.2)
    print(f"  {name_y.mean():>15.1%}   {auc(*curve):.3f}  {t:>6.1%}  {f:>6.1%}   {prec:>8.1%}   {average_precision(*pr):>10.3f}")

# --- Figure: ROC barely moves, precision-recall collapses -------------------------
fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.6))
left.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
left.step(fpr, tpr, where="post", color=BLUE, label=f"{y_test.mean():.0%} default")
left.step(fpr_rare, tpr_rare, where="post", color=ORANGE, label=f"{y_rare.mean():.1%} default")
left.set_title("ROC: almost unchanged")
left.set_xlabel("false positive rate")
left.set_ylabel("true positive rate")
left.legend(loc="lower right")

right.step(recall, precision, where="post", color=BLUE, label=f"{y_test.mean():.0%} default")
right.step(recall_rare, precision_rare, where="post", color=ORANGE, label=f"{y_rare.mean():.1%} default")
right.set_title("Precision-recall: much worse")
right.set_xlabel("recall (= true positive rate)")
right.set_ylabel("precision")
right.legend(loc="upper right")
for ax in (left, right):
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
save(fig, "roc_vs_precision_recall.png")
