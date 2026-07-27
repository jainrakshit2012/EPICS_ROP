import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def hard_voting(probs_matrix):
    """probs_matrix: (n_models, n_samples) of P(class=1). Ties (only possible with an even
    number of models) go to class 0 (Normal), matching argmax-over-vote-sum semantics."""
    votes = (probs_matrix > 0.5).astype(int)
    n_models = probs_matrix.shape[0]
    vote_sum = votes.sum(axis=0)
    return (vote_sum > n_models / 2).astype(int)


def soft_voting(probs_matrix):
    avg_prob = probs_matrix.mean(axis=0)
    return (avg_prob > 0.5).astype(int), avg_prob


def weighted_average_voting(probs_matrix, weights):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    weighted_prob = np.average(probs_matrix, axis=0, weights=weights)
    return (weighted_prob > 0.5).astype(int), weighted_prob


def mix_voting(probs_matrix):
    """Hard Voting, but ties are broken with Soft Voting instead of defaulting to class 0."""
    votes = (probs_matrix > 0.5).astype(int)
    n_models = probs_matrix.shape[0]
    vote_sum = votes.sum(axis=0)
    avg_prob = probs_matrix.mean(axis=0)

    hard_pred = (vote_sum > n_models / 2).astype(int)
    soft_pred = (avg_prob > 0.5).astype(int)
    is_tie = vote_sum == n_models / 2

    return np.where(is_tie, soft_pred, hard_pred)


def compute_metrics(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc"] = float("nan")
    return metrics
