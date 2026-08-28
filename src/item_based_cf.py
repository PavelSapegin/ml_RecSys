from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedCF:

    def __init__(self, df: pd.DataFrame, min_neighbors: int=5, k: int = 50):
        self.k = k
        self.min_neighbors = min_neighbors

        self.user_means_: pd.Series | None = None
        self.user_item_centered_: pd.DataFrame | None = None
        self.item_sim_: pd.DataFrame | None = None

        self._fit(df)


    def _fit(self, df: pd.DataFrame) -> "ItemBasedCF":
        self.user_means_ = df.groupby('userId')['rating'].mean()

        df_centered = df.copy()
        df_centered['rating_centered'] = df_centered['rating'] - \
        df_centered['userId'].map(self.user_means_)

        self.user_item_centered_ = df_centered.pivot(index='userId', columns='movieId', 
                                                values='rating_centered').fillna(0)

        item_sim_matrix = cosine_similarity(self.user_item_centered_.T)
        self.item_sim_ = pd.DataFrame(item_sim_matrix, index=self.user_item_centered_.columns, 
                                columns=self.user_item_centered_.columns)

        return self

    def _predict_rating(self, userId: int, i: int) -> float:
        if self.user_item_centered_ is None or self.item_sim_ is None or self.user_means_ is None:
            raise ValueError("Модель ещё не обучена. Вызовите метод fit().")
        
        if userId not in self.user_item_centered_.index or i not in self.item_sim_.columns:
            return 0.0

        user_row = self.user_item_centered_.loc[userId]
        watched_items = user_row.index[user_row.ne(0).to_numpy()]

        sim_scores_all = self.item_sim_.loc[i]
        sim_scores = sim_scores_all.loc[watched_items]
        sim_scores_pos = cast(pd.Series, sim_scores[sim_scores.gt(0).to_numpy()])
        sim_top_k = sim_scores_pos.sort_values(ascending=False).head(self.k)
        user_ratings = user_row.loc[sim_top_k.index]

        sim_sum = sim_top_k.sum()
        if sim_sum == 0 or len(sim_top_k) < self.min_neighbors:
            return 0.0

        rating_diff = np.dot(sim_top_k, user_ratings) / sim_sum
        final_rating = self.user_means_.loc[userId] + rating_diff
        final_rating = np.clip(final_rating, 0.5, 5.0).item()

        return float(final_rating)


    def recommend_top_n(self, userId: int, top_n: int = 10) -> pd.DataFrame:
        if self.user_item_centered_ is None or self.item_sim_ is None or self.user_means_ is None:
            raise ValueError("Модель ещё не обучена. Вызовите метод fit().")
        
        if userId not in self.user_item_centered_.index:
            raise ValueError(f"Пользователь с userId={userId} отсутствует в обучающих данных.")

        user_vector = self.user_item_centered_.loc[userId]
        unwatched_ids = user_vector[user_vector.eq(0).to_numpy()].index

        predictions = []
        for movieId in unwatched_ids:
            pred_r = self._predict_rating(userId, movieId)
            if pred_r > 0.0:
                predictions.append((movieId, pred_r))

        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:top_n]

        return pd.DataFrame(top_predictions, columns=['movieId', 'predicted_rating'])
