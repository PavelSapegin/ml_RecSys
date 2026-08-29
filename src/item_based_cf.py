from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedCF:

    def __init__(self, df: pd.DataFrame, min_neighbors: int = 1, k: int = 50, beta: int = 10):
        self.k = k
        self.beta = beta
        self.min_neighbors = min_neighbors

        self.user_means_: pd.Series | None = None
        self.user_item_centered_: pd.DataFrame | None = None
        self.item_sim_: pd.DataFrame | None = None

        self._fit(df)

    def _fit(self, df: pd.DataFrame) -> "ItemBasedCF":
        user_item_raw = df.pivot(index="userId", columns="movieId", values="rating")
        self.watched_matrix_ = user_item_raw.notna()
        self.user_means_ = user_item_raw.mean(axis=1)

        user_item_centered_df = user_item_raw.sub(self.user_means_, axis=0)
        self.user_item_centered_ = user_item_centered_df.fillna(0.0)

        item_sim_matrix = cosine_similarity(self.user_item_centered_.T)
        np.fill_diagonal(item_sim_matrix, 0)
        item_sim_matrix[item_sim_matrix < 0] = 0

        watched_int = self.watched_matrix_.astype(int)
        co_counts = np.dot(watched_int.T, watched_int)

        shrinkage_factor = co_counts / (co_counts + self.beta)
        item_sim_matrix = item_sim_matrix * shrinkage_factor
        self.item_sim_ = pd.DataFrame(
            item_sim_matrix,
            index=self.user_item_centered_.columns,
            columns=self.user_item_centered_.columns
        )
        return self

    def _predict_rating(self, userId: int) -> pd.Series:
        if self.user_item_centered_ is None or self.item_sim_ is None or self.user_means_ is None:
            raise ValueError("Модель ещё не обучена. Вызовите метод fit().")
        
        if userId not in self.user_item_centered_.index:
            raise ValueError(f"Пользователь с userId={userId} отсутствует в данных.")

        watched_mask = self.watched_matrix_.loc[userId].to_numpy()

        if not np.any(watched_mask):
            return pd.Series(dtype=float, index=self.user_item_centered_.columns)

        user_ratings_centered = self.user_item_centered_.loc[userId].to_numpy()
        
        sim_matrix = self.item_sim_.iloc[:, watched_mask].to_numpy().copy()
        user_ratings_watched = user_ratings_centered[watched_mask]

        if sim_matrix.shape[1] > self.k:
            sorted_indices = np.argsort(-sim_matrix, axis=1)
            for i in range(sim_matrix.shape[0]):
                sim_matrix[i, sorted_indices[i, self.k:]] = 0.0

        neighbor_counts = (sim_matrix > 0).sum(axis=1)
        sim_sums = sim_matrix.sum(axis=1)
        rating_diffs = np.dot(sim_matrix, user_ratings_watched)

        valid_mask = (sim_sums > 0) & (neighbor_counts >= self.min_neighbors)

        final_ratings = np.zeros(len(self.item_sim_))
        user_mean = self.user_means_.loc[userId]

        final_ratings[valid_mask] = user_mean + (rating_diffs[valid_mask] / sim_sums[valid_mask])
        final_ratings[valid_mask] = np.clip(final_ratings[valid_mask], 0.5, 5.0)

        result = pd.Series(final_ratings, index=self.item_sim_.columns)

        return cast(pd.Series, result[~watched_mask])

    def recommend_top_n(self, userId: int, top_n: int = 10) -> pd.DataFrame:
        preds = self._predict_rating(userId)
        valid_preds = preds[preds > 0].sort_values(ascending=False).head(top_n)
        return valid_preds.reset_index().rename(
            columns={"index": "movieId", 0: "predicted_rating"}
        )
