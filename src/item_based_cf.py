import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedCF:
    def __init__(self, k: int = 20, beta: int = 10, min_ratings: int = 5):
        self.k = k
        self.beta = beta
        self.min_ratings = min_ratings
        self.user_item_df_: pd.DataFrame | None = None
        self.item_sim_df_: pd.DataFrame | None = None
        self.popular_items_: list[int] = []

    def fit(self, train_df: pd.DataFrame) -> "ItemBasedCF":
        self.popular_items_ = train_df["movieId"].value_counts().index.astype(int).tolist()
        item_counts = train_df["movieId"].value_counts()
        valid_items = item_counts[item_counts >= self.min_ratings].index
        filtered_df = train_df[train_df["movieId"].isin(valid_items)]

        self.user_item_df_ = filtered_df.pivot(index="userId", columns="movieId", values="rating")
        filled_df = self.user_item_df_.fillna(0.0)

        sim_matrix = cosine_similarity(filled_df.T)
        np.fill_diagonal(sim_matrix, 0.0)
        watched_mask = self.user_item_df_.notna().astype(int).to_numpy()
        co_counts = np.dot(watched_mask.T, watched_mask)
        shrinkage_factor = co_counts / (co_counts + self.beta)

        sim_matrix = sim_matrix * shrinkage_factor

        self.item_sim_df_ = pd.DataFrame(
            sim_matrix,
            index=self.user_item_df_.columns,
            columns=self.user_item_df_.columns,
        )
        return self

    def recommend_top_n(self, userId: int, top_n: int = 10) -> list[int]:
        # After
        if (
            self.user_item_df_ is None
            or self.item_sim_df_ is None
            or userId not in self.user_item_df_.index
        ):
            return self.popular_items_[:top_n]

        user_ratings = self.user_item_df_.loc[userId].dropna()
        watched_set = set(user_ratings.index)

        if not watched_set:
            return self.popular_items_[:top_n]

        valid_watched = [m for m in watched_set if m in self.item_sim_df_.columns]
        if not valid_watched:
            return self.popular_items_[:top_n]

        sim_sub = self.item_sim_df_.loc[:, valid_watched].to_numpy().copy()
        ratings_vec = user_ratings.loc[valid_watched].to_numpy() - 2.5

        if sim_sub.shape[1] > self.k:
            idx = np.argpartition(sim_sub, -self.k, axis=1)[:, : -self.k]
            rows = np.arange(sim_sub.shape[0])[:, None]
            sim_sub[rows, idx] = 0.0

        scores = np.dot(sim_sub, ratings_vec)
        scores_series = pd.Series(scores, index=self.item_sim_df_.columns)
        scores_series = scores_series.drop(labels=valid_watched, errors="ignore")

        cf_recs = [int(x) for x in scores_series.sort_values(ascending=False).index.tolist()]

        final_recs: list[int] = []
        for item in cf_recs:
            if len(final_recs) == top_n:
                break
            final_recs.append(item)

        if len(final_recs) < top_n:
            for pop_item in self.popular_items_:
                if pop_item not in watched_set and pop_item not in final_recs:
                    final_recs.append(pop_item)
                if len(final_recs) == top_n:
                    break

        return final_recs
