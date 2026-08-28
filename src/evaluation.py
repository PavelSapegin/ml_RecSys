import math


def precision_at_k(recommended: list, relevant: set, k: int) -> float:

    if not relevant or k <= 0:
        return 0
    
    rec_at_k = recommended[:k]
    hits = sum(1 for item in rec_at_k if item in relevant)
    return hits / k

def recall_at_k(recommended: list, relevant: set, k: int) -> float:

    if not relevant or k <= 0:
        return 0
    
    rec_at_k = recommended[:k]
    hits = sum(1 for item in rec_at_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:

    if not relevant or k <= 0:
        return 0
    
    rec_at_k = recommended[:k]

    dcg = 0.0
    for i, item in enumerate(rec_at_k):
        dcg += (item in relevant) / math.log2(i + 2)

    if dcg == 0.0:
        return dcg

    n_relevant_in_top_k = min(len(relevant), k)
    
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant_in_top_k))
    return dcg / idcg


if __name__ == "__main__":
    recommended = ["Film_A", "Film_B", "Film_C", "Film_D", "Film_E"]
    relevant_1_pos = {"Film_A"}
    relevant_not_pos = {"Film_F"}
    relevant_3_pos = {"Film_C"}
    k = 5

    assert abs(precision_at_k(recommended, relevant_1_pos,k) - 0.2) < 1e-3 and \
    abs(recall_at_k(recommended, relevant_1_pos, k) - 1) < 1e-3 and \
    abs(ndcg_at_k(recommended, relevant_1_pos, k) - 1) < 1e-3

    assert abs(precision_at_k(recommended, relevant_not_pos,k) - 0.0) < 1e-3 and \
        abs(recall_at_k(recommended, relevant_not_pos, k) - 0.0) < 1e-3 and \
        abs(ndcg_at_k(recommended, relevant_not_pos, k) - 0.0) < 1e-3
        
    assert abs(precision_at_k(recommended, relevant_3_pos,k) - 0.2) < 1e-3 and \
            abs(recall_at_k(recommended, relevant_3_pos, k) - 1.0) < 1e-3 and \
            abs(ndcg_at_k(recommended, relevant_3_pos, k) - 0.5) < 1e-3
            
    
