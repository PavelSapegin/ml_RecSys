from typing import Any, cast

import pandas as pd


class HybridRecommender:

    def __init__(self,
                 model_a: Any,
                 model_b: Any,
                 movies_df: pd.DataFrame,
                 ratings_df: pd.DataFrame | None = None,
                 alpha: float = 0.5
                 ):

        self.model_a = model_a
        self.model_b = model_b
        self.movies = movies_df
        self.ratings = ratings_df
        self.alpha = alpha


    def _min_max_scale(self, series: pd.Series) -> pd.Series:

        min_val, max_val = series.min(), series.max()

        if max_val == min_val or pd.isna(max_val):
            return series

        scaled = (series - min_val) / (max_val - min_val)
        return cast(pd.Series, scaled)

    def recommend_top_n(self,
                        user_id: int,
                        n: int = 10,
                        filtered_watched: bool = True
                        ) -> pd.DataFrame:


        scores_a = self.model_a.recommend_top_n(
            user_id=user_id,
            n=len(self.movies),
            filtered_watched=False
        )

        df_a = pd.DataFrame(scores_a, columns=["movieId", "score_a"])

        scores_b = self.model_b.recommend_top_n(
            user_id=user_id,
            n=len(self.movies),
            filtered_watched=False
        )

        df_b = pd.DataFrame(scores_b, columns=["movieId", "score_b"])

        combined = pd.merge(df_a, df_b, on="movieId", how="outer")

        combined["norm_a"] = self._min_max_scale(combined["score_a"])
        combined["norm_b"] = self._min_max_scale(combined["score_b"])

        weighted = self.alpha * combined["norm_a"] + (1 - self.alpha) * combined["norm_b"]
        combined["final_score"] = weighted.fillna(combined["norm_a"]).fillna(combined["norm_b"])
        if filtered_watched and self.ratings is not None:
            watched_ids = self.ratings[self.ratings["userId"] == user_id]["movieId"].unique()
            combined = combined[~combined["movieId"].isin(watched_ids)]
            
        recommendations = combined.merge(
            self.movies[["movieId", "title"]],
            on="movieId",
            how="left"
        )

        result_cols = ["movieId", "title", "final_score", "norm_a", "norm_b"]
        return recommendations.sort_values(by="final_score", ascending=False)[result_cols].head(n)
    
