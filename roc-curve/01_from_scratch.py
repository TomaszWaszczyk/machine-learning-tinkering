"""Lesson 1 - the ROC curve and AUC from scratch, in pure Python.

Run it with:

    python3 01_from_scratch.py

No libraries are needed. A classifier gives every example a score. To get a
yes/no answer we cut the scores at a threshold, and every threshold gives a
different mix of mistakes. The ROC curve shows all of those mixes at once.

New ideas in this lesson:

  * true positive rate   - the share of real positives we catch (= recall)
  * false positive rate  - the share of real negatives we wrongly flag
  * the ROC curve        - one (FPR, TPR) point for every possible threshold
  * AUC                  - the area under that curve, a single summary number
  * AUC as a probability - the chance a random positive outranks a random negative
"""

from datasets import TOY_LABELS, TOY_SCORES, make_exam_data


# --- One threshold, one point ------------------------------------------------
def rates(labels, scores, threshold):
    """Return (fpr, tpr) when everything scoring >= threshold is flagged."""
    tp = sum(1 for y, s in zip(labels, scores) if s >= threshold and y == 1)
    fp = sum(1 for y, s in zip(labels, scores) if s >= threshold and y == 0)
    positives = sum(labels)
    negatives = len(labels) - positives
    # TPR: of the real positives, what share did we flag?  (this is recall)
    # FPR: of the real negatives, what share did we flag?  (false alarms)
    return fp / negatives, tp / positives


print("ten borrowers, sorted by the model's score (1 = defaulted)")
print("  score:", "  ".join(f"{s:.2f}" for s in TOY_SCORES))
print("  label:", "  ".join(f"{y:>4}" for y in TOY_LABELS))

print("\nflag everyone with score >= threshold")
print("  threshold   flagged   TPR    FPR")
for threshold in (0.9, 0.75, 0.5, 0.35, 0.05):
    fpr, tpr = rates(TOY_LABELS, TOY_SCORES, threshold)
    flagged = sum(1 for s in TOY_SCORES if s >= threshold)
    print(f"  {threshold:>7.2f}   {flagged:>7}   {tpr:.2f}   {fpr:.2f}")


# --- Every threshold: the ROC curve -----------------------------------------
def roc_curve(labels, scores):
    """Return the ROC curve as a list of (fpr, tpr) points.

    Only thresholds equal to one of the scores matter: anywhere between two
    neighbouring scores the same examples are flagged. So we sort by score,
    highest first, and lower the threshold one example at a time. Each real
    positive we pass moves the curve UP, each real negative moves it RIGHT.
    """
    positives = sum(labels)
    negatives = len(labels) - positives
    ranked = sorted(zip(scores, labels), reverse=True)

    points = [(0.0, 0.0)]  # threshold above every score: nothing is flagged
    tp = fp = 0
    for i, (score, label) in enumerate(ranked):
        if label == 1:
            tp += 1
        else:
            fp += 1
        # Examples with the same score are flagged together, so only add a
        # point once we have passed the whole group of ties.
        is_last_of_ties = i + 1 == len(ranked) or ranked[i + 1][0] != score
        if is_last_of_ties:
            points.append((fp / negatives, tp / positives))
    return points  # always ends at (1, 1): everything is flagged


points = roc_curve(TOY_LABELS, TOY_SCORES)
print("\nthe ROC curve of the ten borrowers, as (FPR, TPR) points")
print("  " + " ".join(f"({fpr:.2f},{tpr:.2f})" for fpr, tpr in points))


# --- The area under the curve -------------------------------------------------
def auc(points):
    """Area under the curve, summing the trapezoid under each segment."""
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2
    return area


def auc_by_pairs(labels, scores):
    """The same number, with no curve at all.

    Take every (positive, negative) pair. Count the share of pairs in which
    the positive got the higher score. A tie counts as half.
    """
    positive_scores = [s for y, s in zip(labels, scores) if y == 1]
    negative_scores = [s for y, s in zip(labels, scores) if y == 0]
    wins = 0.0
    for p in positive_scores:
        for n in negative_scores:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(positive_scores) * len(negative_scores))


print(f"\nAUC as the area under the curve:        {auc(points):.4f}")
print(f"AUC as the share of correctly ordered")
print(f"(defaulter, repayer) pairs:             {auc_by_pairs(TOY_LABELS, TOY_SCORES):.4f}")

# --- Reference points ---------------------------------------------------------
perfect = [1.0 if y == 1 else 0.0 for y in TOY_LABELS]
constant = [0.5] * len(TOY_LABELS)
backwards = [1 - s for s in TOY_SCORES]
print("\nreference points, on the same ten borrowers")
print(f"  perfect scores (all 1s above all 0s):  AUC = {auc(roc_curve(TOY_LABELS, perfect)):.3f}")
print(f"  the same score for everyone:           AUC = {auc(roc_curve(TOY_LABELS, constant)):.3f}")
print(f"  our scores turned upside down:         AUC = {auc(roc_curve(TOY_LABELS, backwards)):.3f}")

# --- A real dataset, and a model that is not a model --------------------------
# ROC needs a score that RANKS the examples; it does not have to be a
# probability. On the exam data, "hours studied" itself is a usable score.
hours, passed = make_exam_data()
exam_points = roc_curve(passed, hours)
print(f"\nexam data: {len(hours)} students, score = hours studied")
print(f"  the curve has {len(exam_points)} points")
print(f"  AUC = {auc(exam_points):.4f}")

# Only the ORDER of the scores matters. Squaring the hours changes every
# score but keeps the order, so the curve and the AUC do not move at all.
squared = [h * h for h in hours]
print(f"  AUC with score = hours squared: {auc(roc_curve(passed, squared)):.4f}")
