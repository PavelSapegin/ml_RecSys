from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:

    def __init__(self, vectorizer_params: dict[str, Any] | None = None) -> None:
        params = vectorizer_params or {}
        self.vectorizer = TfidfVectorizer(**params)

        self.movies_df: pd.DataFrame | None = None

        self.X_tfidf: spmatrix | None = None


        self.user_profiles: dict[Any, np.ndarray] = {}
        self.user_watched: dict[Any, set[Any]] = {}

    def fit(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
        item_col: str = "movieId",
        text_col: str = "genres",
        user_col: str = "userId",
        rating_col: str = "rating",
    ) -> "ContentBasedRecommender":

        self.movies_df = movies.copy().set_index(item_col)
        self.X_tfidf = self.vectorizer.fit_transform(self.movies_df[text_col])

        tfidf_df = pd.DataFrame(
            self.X_tfidf.toarray(), index=self.movies_df.index
        )

        grouped = ratings.groupby(user_col)


        for user_id, user_data in grouped:

            user_df = pd.DataFrame(user_data)

            watched_items = user_df[item_col].tolist()
            self.user_watched[user_id] = set(watched_items)

            valid_user_data = user_df[
                user_df[item_col].isin(self.movies_df.index)
            ]

            if valid_user_data.empty:
                continue

            user_items_ids = valid_user_data[item_col].tolist()
            user_ratings = valid_user_data[rating_col].to_numpy()

            user_item_vectors = tfidf_df.loc[user_items_ids].values

            sum_ratings = user_ratings.sum()
            if sum_ratings > 0:
                profile = (user_ratings @ user_item_vectors) / sum_ratings
            else:
                profile = user_item_vectors.mean(axis=0)

            self.user_profiles[user_id] = profile

        return self

    def recommend_top_n(
        self,
        user_id: Any,
        top_n: int = 10,
        filter_watched: bool = True,
        return_titles_only: bool = True,
        title_col: str = "title",
    ) -> list[str] | pd.Series:


        if self.movies_df is None or self.X_tfidf is None:
            raise RuntimeError(
                "Модель не обучена. Вызовите метод fit() перед рекомендациями."
            )

        if user_id not in self.user_profiles:
            raise ValueError(
                f"Пользователь с user_id={user_id} отсутствует в обученной модели."
            )

        user_profile = self.user_profiles[user_id].reshape(1, -1)

        sim_scores = cosine_similarity(user_profile, self.X_tfidf).flatten()
        rec_scores = pd.Series(sim_scores, index=self.movies_df.index)

        if filter_watched and user_id in self.user_watched:
            watched = self.user_watched[user_id]
            rec_scores = rec_scores.drop(index=list(watched), errors="ignore")

        top_scores = rec_scores.sort_values(ascending=False).head(top_n)

        if return_titles_only:
            return list(
                self.movies_df.loc[top_scores.index][title_col].tolist()
            )

        return top_scores
