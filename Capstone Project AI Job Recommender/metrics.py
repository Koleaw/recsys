# metrics.py
from typing import Sequence, List
import numpy as np


def dcg_at_k(relevances: Sequence[float], k: int = 5) -> float:
    """
    Discounted Cumulative Gain at K.
    relevances должны быть уже отсортированы по предсказанному score.
    """
    rel = np.asfarray(relevances)[:k]
    if rel.size == 0:
        return 0.0

    discounts = np.log2(np.arange(2, rel.size + 2))  # log2(2), log2(3), ...
    gains = (2.0 ** rel - 1.0) / discounts
    return float(np.sum(gains))


def ndcg_at_k(
    true_relevances: Sequence[float],
    predicted_scores: Sequence[float],
    k: int = 5,
) -> float:
    """
    NDCG@K для одного "запроса" (одного кандидата или одной вакансии).
    true_relevances — релевантности (0,1,2,3...),
    predicted_scores — скоры модели для тех же объектов.
    """
    true_relevances = np.asfarray(true_relevances)
    predicted_scores = np.asfarray(predicted_scores)

    if true_relevances.size == 0:
        return 0.0

    # сортировка по предсказанным скором
    order = np.argsort(predicted_scores)[::-1]
    rel_pred = true_relevances[order]
    dcg = dcg_at_k(rel_pred, k=k)

    # идеальный порядок — по релевантности
    ideal_order = np.argsort(true_relevances)[::-1]
    rel_ideal = true_relevances[ideal_order]
    idcg = dcg_at_k(rel_ideal, k=k)

    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def mean_ndcg_at_k(
    all_true: List[Sequence[float]],
    all_scores: List[Sequence[float]],
    k: int = 5,
) -> float:
    """
    Средний NDCG@K по множеству "запросов".
    Например: для каждого кандидата список вакансий с их relevance и score.
    """
    assert len(all_true) == len(all_scores)
    values = [
        ndcg_at_k(true_rel, scores, k=k)
        for true_rel, scores in zip(all_true, all_scores)
    ]
    return float(np.mean(values)) if values else 0.0
