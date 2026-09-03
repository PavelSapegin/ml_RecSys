import math
from typing import Any

import numpy as np
import pandas as pd


def precision_at_k(recommended: list, relevant: set, k: int) -> float:

    if not relevant or k <= 0:
        return 0

    rec_at_k = recommended[:k]
    hits = sum(1 for item in rec_at_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:

    if not relevant or k <= 0:
        return 0

    rec_at_k = recommended[:k]
    hits = sum(1 for item in rec_at_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:

    if not relevant or k <= 0:
        return 0

    rec_at_k = recommended[:k]

    dcg = 0.0
    for i, item in enumerate(rec_at_k):
        dcg += (item in relevant) / math.log2(i + 2)

    if dcg == 0.0:
        return dcg

    n_relevant_in_top_k = min(len(relevant), k)

    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant_in_top_k))
    return dcg / idcg


def evaluate_recommender(
    model: Any,
    test_df: pd.DataFrame,
    k: int = 10,
    rating_threshold: float = 4.0,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
) -> dict[str, float]:

    relevant_test = test_df[test_df[rating_col] >= rating_threshold]

    user_relevant_map = relevant_test.groupby(user_col)[item_col].apply(set).to_dict()

    metrics: dict[str, list] = {"precision": [], "recall": [], "ndcg": []}

    for user_id, relevant_items in user_relevant_map.items():
        if not relevant_items:
            continue
        try:
            recs = model.recommend_top_n(
                user_id=user_id,
                n=k,
                filtered_watched=True,
            )

            if isinstance(recs, pd.DataFrame):
                recommended_items = recs[item_col].tolist()

            elif isinstance(recs, list) and recs and isinstance(recs[0], (tuple, list)):
                recommended_items = [item[0] for item in recs]
            else:
                recommended_items = list(recs)

        except (KeyError, ValueError):
            continue

        p_at_k = precision_at_k(recommended_items, relevant_items, k)
        r_at_k = recall_at_k(recommended_items, relevant_items, k)
        n_at_k = ndcg_at_k(recommended_items, relevant_items, k)

        metrics["precision"].append(p_at_k)
        metrics["recall"].append(r_at_k)
        metrics["ndcg"].append(n_at_k)

    if not metrics["precision"]:
        return {f"Precision@{k}": 0.0, f"Recall@{k}": 0.0, f"NDCG@{k}": 0.0}

    return {
        f"Precision@{k}": float(np.mean(metrics["precision"])),
        f"Recall@{k}": float(np.mean(metrics["recall"])),
        f"NDCG@{k}": float(np.mean(metrics["ndcg"])),
    }


if __name__ == "__main__":
    recommended = ["Film_A", "Film_B", "Film_C", "Film_D", "Film_E"]
    relevant_1_pos = {"Film_A"}
    relevant_not_pos = {"Film_F"}
    relevant_3_pos = {"Film_C"}
    k = 5

    assert (
        abs(precision_at_k(recommended, relevant_1_pos, k) - 0.2) < 1e-3
        and abs(recall_at_k(recommended, relevant_1_pos, k) - 1) < 1e-3
        and abs(ndcg_at_k(recommended, relevant_1_pos, k) - 1) < 1e-3
    )

    assert (
        abs(precision_at_k(recommended, relevant_not_pos, k) - 0.0) < 1e-3
        and abs(recall_at_k(recommended, relevant_not_pos, k) - 0.0) < 1e-3
        and abs(ndcg_at_k(recommended, relevant_not_pos, k) - 0.0) < 1e-3
    )

    assert (
        abs(precision_at_k(recommended, relevant_3_pos, k) - 0.2) < 1e-3
        and abs(recall_at_k(recommended, relevant_3_pos, k) - 1.0) < 1e-3
        and abs(ndcg_at_k(recommended, relevant_3_pos, k) - 0.5) < 1e-3
    )
