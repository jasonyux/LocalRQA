from collections import Counter
from typing import Callable
from rouge import Rouge


rouge = Rouge()


def em(prediction, ground_truth, normalize_fn):
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))


def f1(prediction, ground_truth, normalize_fn):
    prediction_tokens = normalize_fn(prediction).split()
    if len(prediction_tokens) == 0:  # in rare cases, the prediction is empty.
        return 0
    
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def recall(prediction, ground_truth, normalize_fn):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    return 1.0 * num_same / len(ground_truth_tokens)


def precision(prediction, ground_truth, normalize_fn):
    prediction_tokens = normalize_fn(prediction).split()
    if len(prediction_tokens) == 0:  # in rare cases, the prediction is empty.
        return 0
    
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    return 1.0 * num_same / len(prediction_tokens)


def rouge_wrapper(prediction, ground_truth):
    try:
        result = rouge.get_scores(prediction, ground_truth, avg=True)
        return result["rouge-1"]["f"], result["rouge-2"]["f"], result["rouge-l"]["f"]
    except:
        return 0.0, 0.0, 0.0


def f1_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([f1(prediction, gt, normalize_fn) for gt in ground_truths])


def exact_match_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([em(prediction, gt, normalize_fn) for gt in ground_truths])


def rouge_score(prediction, ground_truths):
    ground_truths = [x for x in ground_truths if len(x) > 0]
    if (
        len(prediction) == 0 or len(ground_truths) == 0
    ):  # check if empty prediction or if there is no hypothesis with len > 0
        return 0.0, 0.0, 0.0
    scores = [rouge_wrapper(prediction, gt) for gt in ground_truths]
    rouge1 = max(s[0] for s in scores)
    rouge2 = max(s[1] for s in scores)
    rougel = max(s[2] for s in scores)
    return rouge1, rouge2, rougel