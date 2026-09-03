from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.baseline import PopularityBaseline


class ContentBasedRecommender:

    def __init__(
        self,
        vectorizer_params: dict[str, Any] | None = None,
        popularity_quantile: float = 0.75,
    ) -> None:

        default_params = {"token_pattern": r"(?u)\b\w+\b"}
        if vectorizer_params:
            default_params.update(vectorizer_params)

        self.vectorizer = TfidfVectorizer(**default_params)
        self.popularity_quantile = popularity_quantile

        self.movies_df: pd.DataFrame | None = None
        self.X_tfidf: spmatrix | None = None

        self.user_profiles: dict[Any, np.ndarray] = {}
        self.user_watched: dict[Any, set[Any]] = {}
        self.popularity_baseline: PopularityBaseline | None = None

    def _preprocess_text(self, series: pd.Series) -> pd.Series:
        return (
            series.fillna("unknown")
            .astype(str)
            .str.replace("-", "_", regex=False)
            .str.replace("|", " ", regex=False)
            .str.replace("(no genres listed)", "unknown", regex=False)
        )

    def fit(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
        item_col: str = "movieId",
        text_col: str = "genres",
        user_col: str = "userId",
        rating_col: str = "rating",
    ) -> "ContentBasedRecommender":

        clean_movies = movies.drop_duplicates(subset=[item_col]).copy()
        self.movies_df = clean_movies.set_index(item_col)

        cleaned_text = self._preprocess_text(self.movies_df[text_col])
        self.X_tfidf = self.vectorizer.fit_transform(cleaned_text)

        self.popularity_baseline = PopularityBaseline(
            ratings.rename(columns={item_col: "movieId", rating_col: "rating"}),
            quantile=self.popularity_quantile,
        )


        movie_id_to_row = {mid: idx for idx, mid in enumerate(self.movies_df.index)}

        grouped = ratings.groupby(user_col)

        for user_id, user_data in grouped:
            watched_items = user_data[item_col].tolist()
            self.user_watched[user_id] = set(watched_items)

            valid_user_data = user_data[user_data[item_col].isin(movie_id_to_row)]

            if valid_user_data.empty:
                continue

            user_items_ids = valid_user_data[item_col].tolist()
            user_ratings = valid_user_data[rating_col].to_numpy()


            row_indices = [movie_id_to_row[mid] for mid in user_items_ids]
            user_item_vectors = self.X_tfidf[row_indices].toarray()

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
        n: int = 10,
        filtered_watched: bool = True,
    ) -> list[tuple[Any, float]]:

        if self.movies_df is None or self.X_tfidf is None or self.popularity_baseline is None:
            raise RuntimeError("Модель не обучена. Вызовите метод fit() перед рекомендациями.")

        if user_id not in self.user_profiles:
            fallback_recs = self.popularity_baseline.recommend_top_n(n=n * 2)
            pop_ids = [idx for idx in fallback_recs if idx in self.movies_df.index]

            if filtered_watched and user_id in self.user_watched:
                watched = self.user_watched[user_id]
                pop_ids = [idx for idx in pop_ids if idx not in watched]

            return [(movieId, 0.0) for movieId in pop_ids[:n]]

        user_profile = self.user_profiles[user_id].reshape(1, -1)

        sim_scores = cosine_similarity(user_profile, self.X_tfidf).flatten()
        rec_scores = pd.Series(sim_scores, index=self.movies_df.index)

        if filtered_watched and user_id in self.user_watched:
            watched = self.user_watched[user_id]
            rec_scores = rec_scores.drop(index=list(watched), errors="ignore")

        n = min(n, len(rec_scores))
        top_scores = rec_scores.sort_values(ascending=False).head(n)

        return [(movieId, float(score)) for movieId, score in top_scores.items()]
