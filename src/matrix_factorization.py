import numpy as np
import pandas as pd

from src.baseline import PopularityBaseline


class MatrixFactorization:
    def __init__(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        learning_rate: float = 0.005,
        lm: float = 0.02,
        k: int = 20,
        fallback_model: PopularityBaseline | None = None,
        random_state: int | None = 42,
    ):
        self.train_fit = train
        self.val_fit = val

        unique_users = self.train_fit["userId"].unique()
        unique_items = self.train_fit["movieId"].unique()
        n_users = len(unique_users)
        n_items = len(unique_items)

        self.k = k
        self.learning_rate = learning_rate
        self.lm = lm

        self.rng = np.random.RandomState(random_state)

        scale = 1.0 / np.sqrt(k)
        self.p_matrix = self.rng.normal(0, scale, size=(n_users, k))
        self.q_matrix = self.rng.normal(0, scale, size=(n_items, k))

        self.user_id_to_idx = {ids: idx for idx, ids in enumerate(unique_users)}
        self.item_id_to_idx = {ids: idx for idx, ids in enumerate(unique_items)}
        self.idx_to_item_id = {idx: movie_id for movie_id, idx in self.item_id_to_idx.items()}

        self.user_watched_idx = {}
        for uid, group in self.train_fit.groupby("userId"):
            u_idx = self.user_id_to_idx[uid]
            i_indices = {
                self.item_id_to_idx[mid] for mid in group["movieId"] if mid in self.item_id_to_idx
            }
            self.user_watched_idx[u_idx] = i_indices

        self.all_item_indices = np.arange(n_items)

        self.fallback_model = fallback_model or PopularityBaseline(self.train_fit)

    def _sample_negative(self, user_idx: int) -> int:

        user_positives = self.user_watched_idx[user_idx]

        j_idx = self.rng.choice(self.all_item_indices)
        while j_idx in user_positives:
            j_idx = self.rng.choice(self.all_item_indices)

        return int(j_idx)

    def fit(self, n_negatives: int = 5) -> "MatrixFactorization":
        best_val_loss = np.inf
        epochs = 100
        patience_counter = 0
        patience = 7

        p_matrix_best = self.p_matrix.copy()
        q_matrix_best = self.q_matrix.copy()

        u_indices = np.array([self.user_id_to_idx[uid] for uid in self.train_fit["userId"]])
        i_indices = np.array([self.item_id_to_idx[mid] for mid in self.train_fit["movieId"]])
        n_samples = len(u_indices)

        val_mask = [
            (row.userId in self.user_id_to_idx) and (row.movieId in self.item_id_to_idx)
            for row in self.val_fit.itertuples()
        ]
        val_filtered = self.val_fit[val_mask]
        val_u = np.array([self.user_id_to_idx[uid] for uid in val_filtered["userId"]])
        val_i = np.array([self.item_id_to_idx[mid] for mid in val_filtered["movieId"]])

        for epoch in range(epochs):
            perm = self.rng.permutation(n_samples)
            bpr_loss = 0.0

            for idx in perm:
                u = u_indices[idx]
                i = i_indices[idx]

                for _ in range(n_negatives):
                    j = self._sample_negative(u)

                    p_u = self.p_matrix[u]
                    q_i = self.q_matrix[i]
                    q_j = self.q_matrix[j]

                    x_uij = np.dot(p_u, q_i - q_j)

                    bpr_loss += np.log1p(np.exp(-x_uij))

                    sigmoid_grad = 1.0 / (1.0 + np.exp(x_uij))

                    p_u_old = p_u.copy()
                    q_i_old = q_i.copy()
                    q_j_old = q_j.copy()

                    self.p_matrix[u] += self.learning_rate * (
                        sigmoid_grad * (q_i_old - q_j_old) - self.lm * p_u_old
                    )
                    self.q_matrix[i] += self.learning_rate * (
                        sigmoid_grad * p_u_old - self.lm * q_i_old
                    )
                    self.q_matrix[j] += self.learning_rate * (
                        -sigmoid_grad * p_u_old - self.lm * q_j_old
                    )
            train_loss = bpr_loss / (n_samples * n_negatives)

            if len(val_u) > 0:
                val_j = np.array([self._sample_negative(u) for u in val_u])

                val_x_uij = np.sum(
                    self.p_matrix[val_u] * (self.q_matrix[val_i] - self.q_matrix[val_j]), axis=1
                )
                val_loss = np.mean(np.log1p(np.exp(-val_x_uij)))
            else:
                val_loss = 0.0

            print(
f"[{epoch + 1:02d}/{epochs}] Train BPR Loss: {train_loss:.4f} | Val BPR Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                p_matrix_best = self.p_matrix.copy()
                q_matrix_best = self.q_matrix.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping at epoch {epoch + 1}. Best Val Loss: {best_val_loss:.4f}"
                    )
                    break

        self.p_matrix = p_matrix_best
        self.q_matrix = q_matrix_best

        return self

    def recommend_top_n(self, user_id: int, n: int = 10, filtered_watched: bool = True) -> list:
        if user_id not in self.user_id_to_idx:
            fallback_recs = self.fallback_model.recommend_top_n(n=n)
            return [(movieId, 0.0) for movieId in fallback_recs]

        idx = self.user_id_to_idx[user_id]
        user_vector = self.p_matrix[idx]

        pred_ratings = np.dot(self.q_matrix, user_vector)

        if filtered_watched:
            watched_indices = list(self.user_watched_idx.get(idx, set()))
            pred_ratings[watched_indices] = -np.inf

        n = min(n, len(pred_ratings))

        if n == len(pred_ratings):
            top_n_indices = np.argsort(pred_ratings)[::-1]
        else:
                
            top_n_indices = np.argpartition(pred_ratings, -n)[-n:]
            top_n_indices = top_n_indices[np.argsort(pred_ratings[top_n_indices])][::-1]

        recommendations = [(self.idx_to_item_id[i], float(pred_ratings[i])) for i in top_n_indices]
    
        return recommendations

    