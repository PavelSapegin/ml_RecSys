import pandas as pd


class PopularityBaseline:

    def __init__(self, df: pd.DataFrame, quantile: float = 0.75):
        self.quantile = quantile
        self._ranked_movies = self._fit(df)

    def _bayes_weighted_score(self, df_stats: pd.DataFrame, m: float, C: float) -> pd.Series:
        v = df_stats["v"]
        R = df_stats["R"]

        return v*R/(v + m) + m*C/(v + m)
    
    def _fit(self,df: pd.DataFrame) -> list:

        movie_stats = df.groupby("movieId")["rating"].agg(
            v="count",
            R="mean"
        ).reset_index()

        m = movie_stats["v"].quantile(0.75)
        C = df["rating"].mean()

        movie_stats["weighted_rating"] = self._bayes_weighted_score(df_stats=movie_stats, m=m, C=C)
        movie_stats = movie_stats.sort_values(by='weighted_rating', ascending=False)

        return movie_stats["movieId"].to_list()

    def recommend(self) -> list:
        return self._ranked_movies
    
    def recommend_top_n(self, n: int) -> list:
        if n < 1:
            raise ValueError("N must be natural.")

        return self._ranked_movies[:n]
