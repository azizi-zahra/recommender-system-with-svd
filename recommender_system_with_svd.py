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
    return np.sqrt(mean_squared_error(test["rating"], preds))


def extract_year(s):
    if not isinstance(s, str): return np.nan
    m = re.search(r"\((\d{4})\)", s)
    return int(m.group(1)) if m else np.nan


def power_iteration(A, max_iter=1000, tol=1e-6, random_state=None):
    """
    Power iteration for the dominant eigenpair of symmetric matrix A.
    Returns: eigenvalue (lambda), eigenvector (unit norm)
    """
    n = A.shape[0]
    rng = np.random.default_rng(random_state)
    v = rng.random(n)
    v = v / np.linalg.norm(v)

    for _ in range(max_iter):
        Av = A.dot(v)
        norm_av = np.linalg.norm(Av)
        if norm_av == 0:
            return 0.0, v
        v_new = Av / norm_av
        if np.linalg.norm(v_new - v) < tol:
            v = v_new
            break
        v = v_new

    # Rayleigh quotient for eigenvalue
    eig_val = v.dot(A.dot(v))
    return eig_val, v


def svd_with_deflation(A, num_singular_values=10, max_iter=1000, tol=1e-6, random_state=None):
    """
    Compute top `num_singular_values` singular triplets of A (m x n) using 
    power iteration on A.T @ A with deflation.
    """
    m, n = A.shape
    ATA = A.T.dot(A)  

    r = min(num_singular_values, n)
    eig_vals = []
    eig_vecs = []

    ATA_work = ATA.copy()
    for i in range(r):
        eig_val, eig_vec = power_iteration(ATA_work, max_iter=max_iter, tol=tol, random_state=(None if random_state is None else random_state + i))
        # If eigenvalue is non-positive (numerical), stop early
        if eig_val <= 0:
            break
        eig_vals.append(eig_val)
        eig_vecs.append(eig_vec)

        # Deflate: subtract λ * v v^T from ATA_work
        ATA_work = ATA_work - eig_val * np.outer(eig_vec, eig_vec)

    if len(eig_vals) == 0:
        return np.zeros((m, 0)), np.array([]), np.zeros((0, n))

    eig_vals = np.array(eig_vals)           
    eig_vecs = np.column_stack(eig_vecs)   

    sorted_idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sorted_idx]
    eig_vecs = eig_vecs[:, sorted_idx]

    s = np.sqrt(eig_vals)  

    V = eig_vecs            
    Vt = V.T

    nonzero_mask = s > 1e-12
    if not np.all(nonzero_mask):
        # trim out any zero singulars (shouldn't normally happen because we stopped on <=0)
        s = s[nonzero_mask]
        V = V[:, nonzero_mask]
        Vt = V.T

    U = A.dot(V) / s[np.newaxis, :]   

    return U, s, Vt

# ===========================

train_df = pd.read_csv("data.csv")
test_df  = pd.read_csv("test.csv")

ratings_matrix = train_df.pivot_table(index="item_id", columns="user_id", values="rating")

# Grid search for finding best k and lambda
lambda_values = [5, 10, 15]
k_values = list(range(5, 16))

best = {"rmse": float("inf"), "k": None, "lambda": None, 
        "U": None, "s":None, "Vt": None, "bias_matrix": None}

for lam in lambda_values:
    global_mean, item_bias, user_bias = calculate_biases_with_shrinkage(ratings_matrix, lam)
    R_detrend, bias_matrix = detrend_matrix(ratings_matrix, global_mean, item_bias, user_bias)

    A = R_detrend.to_numpy()
    U, s, Vt = svd_with_deflation(A, num_singular_values=100, max_iter=1000, tol=1e-6, random_state=42)
    # U, s, Vt = np.linalg.svd(R_detrend.values) # Used for verifying

    for k in k_values:
        rmse = rmse_for_k(U, s, Vt, k, bias_matrix, ratings_matrix.index, ratings_matrix.columns, test_df)
        print(f"lambda={lam:>2}, k={k:>2} --> RMSE={rmse:.4f}")
        if rmse < best["rmse"]:
            best.update({"rmse": rmse, "k": k, "lambda": lam, "U": U, "s": s, "Vt": Vt, "bias_matrix": bias_matrix})
  
print(f'\nBest Result: lambda={best["lambda"]}, k={best["k"]}, RMSE={best["rmse"]:.4f}')

"""
Questions:
1 - Q can be movies genres and themes, P^T can be how much a user 
    like a genre or theme (the latent factors discovered in Q)
2 - Solutions can be filling the missing values with global mean,
    bias-baseed estinates (Start with the global average rating.
    Adjust it up/down for each movie (movie bias) and each user 
    (user bias).), mean of user rates, mean of items rates 
    and similar. I use bias-adjusted SVD with shrinkage This lets 
    the model learn reliable patterns between users and movies from
    the available ratings.
"""

# Enrgy, cumulative variance and RMSE vs K plots 
U, s, Vt = best["U"], best["s"], best["Vt"]
bias_matrix = best["bias_matrix"]

energy = s ** 2
variance_ratio = energy / energy.sum() # The contribution of each component is σ_i² / total_energy
cumulative_variance = np.cumsum(variance_ratio)

rmse = []
k_range = list(range(1, min(80, len(s)) + 1))
for k in k_range:
    rmse_k = rmse_for_k(U, s, Vt, k, bias_matrix, ratings_matrix.index,
                         ratings_matrix.columns, test_df)
    rmse.append(rmse_k)

# Energy
top_n = min(20, len(energy))           
inds = range(1, top_n + 1)

plt.figure(figsize=(10, 4))
plt.bar(inds, energy[:top_n], edgecolor='black')
plt.title("Energy per component")
plt.xlabel("Component")
plt.ylabel("Energy")
step = max(1, top_n // 10) # avoid overcrowding ticks
plt.xticks(list(inds)[::step])

threshold_pct = 0.5  # annotate components that contribute >= 0.5% of total energy
total = energy.sum()
for i, val in enumerate(energy[:top_n]):
    if val / total * 100 >= threshold_pct:
        plt.text(i+1, val + 0.01 * max(energy[:top_n]), f"{val:.1f}", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("Energy_per_component")
plt.show()

# Cumulative variance
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.title("Cumulative variance")
plt.xlabel("Cumulative variance ration")
plt.ylabel("Componenet")
plt.grid(True, alpha=0.3)
plt.savefig("Cumulative_variance")
plt.show()

# RMSE vs K
plt.figure(figsize=(6, 4))
plt.plot(k_range, rmse)
plt.title("RMSE vs K")
plt.xlabel("k")
plt.ylabel("RMSE")
plt.grid(True, alpha=0.3)
plt.savefig("RMSE_vs_K")
plt.show()
