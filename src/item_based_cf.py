import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedCF:

  def __init__(
      self,
      df: pd.DataFrame,
      k: int = 20,
      beta: int = 10,
      min_movie_ratings: int = 5,
  ):
    self.k = k
    self.beta = beta
    self.min_movie_ratings = min_movie_ratings

    self.user_item_raw_: pd.DataFrame | None = None
    self.item_sim_: pd.DataFrame | None = None

    self._fit(df)

  def _fit(self, df: pd.DataFrame) -> "ItemBasedCF":
    movie_counts = df["movieId"].value_counts()
    valid_movies = movie_counts[movie_counts >= self.min_movie_ratings].index
    df_filtered = df[df["movieId"].isin(valid_movies)]

    self.user_item_raw_ = df_filtered.pivot(
        index="userId", columns="movieId", values="rating"
    )
    user_item_filled = self.user_item_raw_.fillna(0.0)

    item_sim_matrix = cosine_similarity(user_item_filled.T)
    np.fill_diagonal(item_sim_matrix, 0)


    watched_mask = self.user_item_raw_.notna().astype(int)
    co_counts = np.dot(watched_mask.T, watched_mask)
    shrinkage_factor = co_counts / (co_counts + self.beta)
    item_sim_matrix = item_sim_matrix * shrinkage_factor

    self.item_sim_ = pd.DataFrame(
        item_sim_matrix,
        index=self.user_item_raw_.columns,
        columns=self.user_item_raw_.columns,
    )
    return self

  def _predict_rating(self, userId: int) -> pd.Series:
        if userId not in self.user_item_raw_.index:
            return pd.Series(dtype=float)

        user_ratings = self.user_item_raw_.loc[userId].dropna()
        valid_watched = [m for m in user_ratings.index if m in self.item_sim_.columns]

        if not valid_watched:
            return pd.Series(dtype=float)

        sim_sub = self.item_sim_.loc[:, valid_watched].to_numpy().copy()
        user_ratings_vec = user_ratings.loc[valid_watched].to_numpy()

        if sim_sub.shape[1] > self.k:
            idx = np.argpartition(sim_sub, -self.k, axis=1)[:, :-self.k]
            rows = np.arange(sim_sub.shape[0])[:, None]
            sim_sub[rows, idx] = 0.0


        user_centered = user_ratings_vec - 2.5
        final_scores = np.dot(sim_sub, user_centered)

        scores_series = pd.Series(final_scores, index=self.item_sim_.columns)

        return scores_series.drop(labels=valid_watched, errors='ignore')

  def recommend_top_n(self, userId: int, top_n: int = 10) -> list[int]:
    preds = self._predict_rating(userId)
    if preds.empty:
      return []


    return preds.sort_values(ascending=False).head(top_n).index.tolist()
