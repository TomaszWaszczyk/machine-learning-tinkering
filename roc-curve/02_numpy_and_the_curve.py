"""Lesson 2 - the ROC curve of a real model, with numpy and matplotlib.

Run it with:

    python 02_numpy_and_the_curve.py

We take the loan-default model from the logistic regression tutorial and
draw its ROC curve. Then we compare it with a weaker model and with models
that know nothing or everything.

New ideas in this lesson:

  * where the curve comes from - two overlapping piles of scores
  * a threshold is a point     - each row of the old threshold table sits on the curve
  * comparing models           - the curve nearer the top-left corner wins
  * the diagonal               - what a model that learned nothing looks like

It saves three figures into images/.
"""

import numpy as np

from loan_model import loan_test_scores
from plotting import BLUE, INK, MUTED, ORANGE, plt, save

y_test, p_test = loan_test_scores()
print(f"{len(y_test)} test borrowers, {y_test.sum()} defaulted ({y_test.mean():.1%})\n")


# --- The ROC curve, vectorised -------------------------------------------------
def roc_curve(y, scores):
    """Return (fpr, tpr, thresholds), one entry per distinct score.

    The same walk as in lesson 1, without the loop: sort by score, highest
    first, and the running totals of positives and negatives passed so far
    ARE the true and false positive counts at each threshold.
    """
    order = np.argsort(-scores, kind="stable")
    y_sorted, s_sorted = y[order], scores[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    # Tied scores are flagged together: keep only the last one of each group.
    keep = np.append(np.diff(s_sorted) != 0, True)
    tpr = np.concatenate([[0.0], tp[keep] / tp[-1]])
    fpr = np.concatenate([[0.0], fp[keep] / fp[-1]])
    thresholds = np.concatenate([[np.inf], s_sorted[keep]])
    return fpr, tpr, thresholds


def auc(fpr, tpr):
    """Area under the curve by the trapezoid rule."""
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


fpr, tpr, thresholds = roc_curve(y_test, p_test)

# --- Each threshold is one point on the curve ---------------------------------
# These are the thresholds from the table in the logistic regression tutorial.
# There we read precision and recall off each row. Recall is the TPR; the new
# column is the FPR, the share of good customers wrongly flagged.
marked = (0.8, 0.5, 0.3, 0.2, 0.1)
print("the same model at different thresholds")
print("  threshold   flagged   TPR (recall)     FPR")
marked_points = []
for threshold in marked:
    flagged = p_test >= threshold
    point_tpr = (flagged & (y_test == 1)).sum() / (y_test == 1).sum()
    point_fpr = (flagged & (y_test == 0)).sum() / (y_test == 0).sum()
    marked_points.append((point_fpr, point_tpr))
    print(f"  {threshold:>7.1f}   {flagged.sum():>7}   {point_tpr:>10.1%}   {point_fpr:>7.1%}")

print(f"\nAUC of the model: {auc(fpr, tpr):.3f}")

# --- Comparing models -----------------------------------------------------------
# A weaker model: the same algorithm, but it is not shown the two most
# informative features (income and missed payments).
y_weak, p_weak = loan_test_scores(features=["debt_k_pln", "age_years"])
fpr_weak, tpr_weak, _ = roc_curve(y_weak, p_weak)

# A model that learned nothing: scores drawn at random. Its AUC will not be
# exactly 0.5: with only 59 defaults in the test set, luck alone moves it.
p_random = np.random.default_rng(1).random(len(y_test))
fpr_random, tpr_random, _ = roc_curve(y_test, p_random)

# A model that knows the answers.
fpr_perfect, tpr_perfect, _ = roc_curve(y_test, y_test.astype(float))

print("\ncomparing models on the same test set")
print(f"  perfect (knows the answers)       AUC = {auc(fpr_perfect, tpr_perfect):.3f}")
print(f"  all four features                 AUC = {auc(fpr, tpr):.3f}")
print(f"  debt and age only                 AUC = {auc(fpr_weak, tpr_weak):.3f}")
print(f"  random scores                     AUC = {auc(fpr_random, tpr_random):.3f}")

# AUC only looks at the order of the scores. Any transformation that keeps
# the order (here: the log-odds instead of the probability) leaves it unchanged.
log_odds = np.log(p_test / (1 - p_test))
print(f"\nAUC using log-odds as the score:    {auc(*roc_curve(y_test, log_odds)[:2]):.3f}")

# --- Figure 1: two piles of scores ----------------------------------------------
bins = np.linspace(0, 1, 26)
fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
top.hist(p_test[y_test == 0], bins=bins, color=BLUE)
top.set_title("The scores of the two classes overlap")
top.set_ylabel("repaid")
bottom.hist(p_test[y_test == 1], bins=bins, color=ORANGE)
bottom.set_ylabel("defaulted")
bottom.set_xlabel("predicted probability of default")
for ax in (top, bottom):
    ax.axvline(0.3, color=INK, linewidth=1, linestyle="--")
top.annotate("threshold 0.3\nflagged: false alarms (FPR)", xy=(0.3, 0.6), xycoords=("data", "axes fraction"),
             xytext=(8, 0), textcoords="offset points", color=MUTED, va="center")
bottom.annotate("flagged: defaults caught (TPR)", xy=(0.3, 0.8), xycoords=("data", "axes fraction"),
                xytext=(8, 0), textcoords="offset points", color=MUTED, va="center")
save(fig, "score_distributions.png")


# --- Figure 2: the ROC curve, with thresholds marked ----------------------------
def roc_axes(title):
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("false positive rate (share of repayers flagged)")
    ax.set_ylabel("true positive rate (share of defaults caught)")
    return fig, ax


fig, ax = roc_axes("Every threshold is one point on the ROC curve")
ax.fill_between(fpr, tpr, step="post", color=BLUE, alpha=0.12, linewidth=0)
ax.step(fpr, tpr, where="post", color=BLUE)
for threshold, (x, y) in zip(marked, marked_points):
    ax.plot(x, y, "o", color=ORANGE, markersize=6, zorder=3)
    ax.annotate(f"{threshold}", xy=(x, y), xytext=(8, -12), textcoords="offset points", color=INK)
ax.text(0.55, 0.3, f"AUC = {auc(fpr, tpr):.2f}", color=BLUE, fontsize=12, fontweight="bold")
ax.text(0.62, 0.56, "random guessing", color=MUTED, rotation=45, ha="center", va="center")
save(fig, "roc_curve.png")

# --- Figure 3: comparing models ---------------------------------------------------
fig, ax = roc_axes("A better model bends towards the top-left")
ax.step(fpr_perfect, tpr_perfect, where="post", color=INK, linewidth=1, linestyle="--",
        label=f"perfect (AUC {auc(fpr_perfect, tpr_perfect):.2f})")
ax.step(fpr, tpr, where="post", color=BLUE, label=f"all four features (AUC {auc(fpr, tpr):.2f})")
ax.step(fpr_weak, tpr_weak, where="post", color=ORANGE,
        label=f"debt and age only (AUC {auc(fpr_weak, tpr_weak):.2f})")
ax.step(fpr_random, tpr_random, where="post", color=MUTED, linewidth=1,
        label=f"random scores (AUC {auc(fpr_random, tpr_random):.2f})")
ax.legend(loc="lower right")
save(fig, "comparing_models.png")
