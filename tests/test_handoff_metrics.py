"""Unit tests for the metric formulas in eval/handoff_eval.py. No API calls."""
from eval.handoff_eval import confusion, majority, ratio, scores

# 3 TP, 1 FN, 2 FP, 4 TN
PAIRS = [(True, True)] * 3 + [(True, False)] + [(False, True)] * 2 + [(False, False)] * 4


def test_confusion_counts():
    assert confusion(PAIRS) == (3, 2, 1, 4)


def test_precision_recall_f1():
    s = scores(PAIRS)
    assert s.precision == 3 / 5 and s.recall == 3 / 4
    assert round(s.f1, 4) == round(2 * 0.6 * 0.75 / (0.6 + 0.75), 4)
    assert (s.p_text, s.r_text) == ("0.60 (3/5)", "0.75 (3/4)")


def test_no_predicted_positives_gives_undefined_precision():
    s = scores([(True, False), (False, False)])
    assert s.precision is None and s.recall == 0 and s.f1 is None
    assert s.p_text == "n/a (0/0)"


def test_majority_of_three():
    assert majority([True, True, False]) and not majority([True, False, False])


def test_ratio_keeps_counts():
    assert ratio(9, 10) == "0.90 (9/10)"
