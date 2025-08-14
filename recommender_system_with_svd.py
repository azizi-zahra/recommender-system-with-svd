import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# ======== Utilities ========
def calculate_biases_with_shrinkage(ratings_matrix, lambda_value):
    """
    r_ui = μ + b_u + b_i + q_i^T p_u 
    If an item has very few ratings, that mean can be noisy.
    Shrinkage blends it toward 0 (no bias) unless the sample size is large:

    Item bias:
        b_i = (|R_i| / (|R_i| + λ)) * (sum_{u ∈ R_i} (r_ui - μ) / |R_i|)

    User bias:
        b_u = (|R_u| / (|R_u| + λ)) * (sum_{i ∈ R_u} (r_ui - μ) / |R_u|)

    Where:
        μ     = global mean rating
        R_i   = set of users who rated item i
        R_u   = set of items rated by user u
        λ     = shrinkage parameter (regularization)
        r_ui  = rating given by user u to item i

    """
    global_mean = ratings_matrix.stack().mean()
    item_counts = ratings_matrix.count(axis=1)
    user_counts = ratings_matrix.count(axis=0)
    item_bias =  (ratings_matrix.mean(axis=1) - global_mean) * (item_counts / (item_counts + lambda_value))
    user_bias =  (ratings_matrix.mean(axis=0) - global_mean) * (user_counts / (user_counts + lambda_value))
    return global_mean, item_bias, user_bias


def detrend_matrix(ratings_matrix, global_mean, item_bias, user_bias):
    bias_matrix = (global_mean + item_bias.values.reshape(-1, 1) + user_bias.values.reshape(1, -1))
    R_detrended = (ratings_matrix - bias_matrix).fillna(0)
    return R_detrended, bias_matrix


def rmse_for_k(U, s, Vt, k, bias_matrix, R_index, R_columns, test):
    Uk = U[:, :k]
    Sk = np.diag(s[:k])
    Vtk = Vt[:k, :]
    R_k = np.dot(Uk, np.dot(Sk, Vtk))
    R_pred = R_k + bias_matrix

    R_pred_df = pd.DataFrame(R_pred, index=R_index, columns=R_columns).clip(1, 5) # Ratings are in range [1, 5]
    preds = test.apply(lambda r: R_pred_df.loc[r["item_id"], r["user_id"]], axis=1)
    return np.sqrt(mean_squared_error(test["retings"], preds))


def extract_year(s):
    if not isinstance(s, str): return np.nan
    m = re.search(r"\((\d{4})\)", s)
    return int(m.group(1)) if m else np.nan

# ===========================

# Load data
train_df = pd.read_csv("data.csv")
test_df  = pd.read_csv("test.csv")

ratings_matrix = train_df.pivot_table(index="item_id", columns="user_id", values="rating")

# Grid search for finding best k and lambda
lambda_values = [5, 10, 15, 20]
k_values = list(range(5, 16))

best = {"rmse": float("inf"), "k": None, "lambda": None, 
        "U": None, "s":None, "Vt": None, "bias_matrix": None}

for lam in lambda_values:
    global_mean, item_bias, user_bias = calculate_biases_with_shrinkage(ratings_matrix, lam)
    R_detrend, bias_matrix = detrend_matrix(ratings_matrix, global_mean, item_bias, user_bias)

    U, s, Vt = np.linalg.svd(R_detrend.values, full_matrices=False) # TODO

    for k in k_values:
        rmse = rmse_for_k(U, s, Vt, k, bias_matrix, ratings_matrix.index, ratings_matrix.columns, test_df)
        print(f"lambda={lam:>2}, k={k:>2} --> RMSE={rmse:.4f}")
        if rmse < best["rmse"]:
            best.update({"rmse": rmse, "k": k, "lambda": lam, "U": U, "s": s, "Vt": Vt, "bias_matrix": bias_matrix})

  
print(f"\nBest Result: lambda={best["lambda"]}, k={best["k"]}, RMSE={best["rmse"]:.4f}")


