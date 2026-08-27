import pandas as pd


def leave_k_last(df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = df.sort_values(
        by=["userId", "timestamp"], 
        ascending=[True, False]
    ).reset_index(drop=True)

    sorted_df["item_rank"] = sorted_df.groupby("userId").cumcount()

    train = sorted_df[sorted_df["item_rank"] >= k].drop(columns=["item_rank"])
    test = sorted_df[sorted_df["item_rank"] < k].drop(columns=["item_rank"])

    return train, test
