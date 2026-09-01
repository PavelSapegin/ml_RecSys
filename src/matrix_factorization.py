import numpy as np
import pandas as pd

from src.baseline import PopularityBaseline
from src.preprocessing import leave_k_last


class MatrixFactorization:

    def __init__(self, df: pd.DataFrame, learning_rate: float =1e-2, lm: float = 1e-2, 
                 k: int = 100, fallback_model: PopularityBaseline | None = None):
        self.df = df
        self.train_fit, self.val_fit = leave_k_last(self.df, k=1)
        
        unique_users = self.train_fit["userId"].unique()
        unique_items = self.train_fit["movieId"].unique()
        n_users = len(unique_users)
        n_items = len(unique_items)

        self.k = k
        self.learning_rate = learning_rate
        self.lm = lm
        self.mu = self.train_fit["rating"].mean()
        self.bias_user = np.zeros(n_users)
        self.bias_item = np.zeros(n_items)
        self.p_matrix = np.random.normal(0, 0.01, size=(n_users, k))
        self.q_matrix = np.random.normal(0, 0.01, size=(n_items, k))

        self.user_id_to_idx = {ids: idx for idx, ids in enumerate(unique_users)}
        self.item_id_to_idx = {ids: idx for idx, ids in enumerate(unique_items)}
        self.idx_to_item_id = {idx: movie_id for movie_id, idx in self.item_id_to_idx.items()}
        self.user_watched = self.df.groupby("userId")["movieId"].apply(set).to_dict()

        self.fallback_model = fallback_model or PopularityBaseline(self.train_fit)

    def fit(self,) -> "MatrixFactorization":

        
        best_val_loss = float("inf")
        best_epoch = 0
        epochs = 100
        patience_counter = 0
        patience = 10

        p_matrix_best = self.p_matrix.copy()
        q_matrix_best = self.q_matrix.copy()
        bias_user_best = self.bias_user.copy()
        bias_item_best = self.bias_item.copy()

        for epoch in range(epochs):

            shuffled_train = self.train_fit.sample(frac=1, random_state=42 + epoch)

            train_loss = 0.0
            squared_error = 0.0
            for row in shuffled_train.itertuples():

                u = self.user_id_to_idx[row.userId]
                i = self.item_id_to_idx[row.movieId]
                r = row.rating

                rating_pred = self.mu + self.bias_user[u] + self.bias_item[i] + self.p_matrix[u] @ \
                self.q_matrix[i]
                error = r - rating_pred

                p_old = self.p_matrix[u].copy()
                q_old = self.q_matrix[i].copy()
                self.p_matrix[u] += self.learning_rate * (error * q_old - self.lm * p_old)
                self.q_matrix[i] += self.learning_rate * (error * p_old - self.lm * q_old)
                self.bias_user[u] += self.learning_rate * (error - self.lm * self.bias_user[u])
                self.bias_item[i] += self.learning_rate * (error - self.lm * self.bias_item[i])

                squared_error += error**2

            train_loss = np.sqrt(squared_error/len(shuffled_train))

            squared_error = 0.0
            val_loss = 0.0
            skipped_ids = 0
            for row in self.val_fit.itertuples():

                if row.userId not in self.user_id_to_idx or row.movieId not in self.item_id_to_idx:
                    skipped_ids += 1
                    continue

                u = self.user_id_to_idx[row.userId]
                i = self.item_id_to_idx[row.movieId]
                r = row.rating

                rating_pred = self.mu + self.bias_user[u] + self.bias_item[i] + self.p_matrix[u] @ \
                self.q_matrix[i]
                squared_error += (r - rating_pred)**2

            val_count = (len(self.val_fit) - skipped_ids)
            val_loss = np.sqrt(squared_error/val_count) if val_count > 0 else float("inf") 

            print(f"[{epoch}/{epochs}] Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                p_matrix_best = self.p_matrix.copy()
                q_matrix_best = self.q_matrix.copy()
                bias_user_best = self.bias_user.copy()
                bias_item_best = self.bias_item.copy()

                patience_counter = 0

            else:
                patience_counter += 1

                if patience_counter == patience:
                    best_epoch = epoch
                    print(f"[Final best epoch: {best_epoch}], Train loss: {train_loss:.4f}, \
                          Val loss: {val_loss:.5f}")
                    break


        self.p_matrix = p_matrix_best
        self.q_matrix = q_matrix_best
        self.bias_user = bias_user_best
        self.bias_item = bias_item_best

        return self


    def recommend_top_n(self, userId: int, n:int) -> list:

        if userId not in self.user_id_to_idx:
            fallback_recs = self.fallback_model.recommend_top_n(n=n)
            return [(movie_id, self.mu) for movie_id in fallback_recs]

        idx = self.user_id_to_idx[userId]

        user_vector = self.p_matrix[idx]

        pred_ratings = self.mu + self.bias_user[idx] + self.bias_item + user_vector @ \
        self.q_matrix.T

        watched_movie_ids = self.user_watched.get(userId, set())
        watched_indices = [self.item_id_to_idx[m_id] for m_id in watched_movie_ids 
                           if m_id in self.item_id_to_idx]

        pred_ratings[watched_indices] = -np.inf
        top_n_indices = np.argsort(pred_ratings)[::-1][:n]

        recommendations = [(self.idx_to_item_id[idx], float(pred_ratings[idx])) 
                           for idx in top_n_indices]

        return recommendations
