import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_distances
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


def pca_transform(X, n_components=2):
    X_centered = X - np.mean(X, axis=0)
    
    # Covariance matrix
    cov = np.cov(X_centered, rowvar=False)
    
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    
    # Fix sign convention (to have similar result with sklearn PCA)
    for i in range(eigvecs.shape[1]):
        if np.abs(eigvecs[:, i]).max() > 0:
            if eigvecs[np.argmax(np.abs(eigvecs[:, i])), i] < 0:
                eigvecs[:, i] *= -1
    
    W = eigvecs[:, :n_components]
    return X_centered @ W

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

# Interpretation of hidden factors
k_best = best["k"]

# Movies
movies_meta = train_df.groupby("item_id", as_index=False)["rating"].mean()
movies_meta = movies_meta.rename(columns={"rating": "avg_rating"})

titles = train_df[["item_id", "title"]].dropna().drop_duplicates("item_id")
movies_meta = movies_meta.merge(titles, on="item_id", how="left")
movies_meta["year"] = movies_meta["title"].apply(extract_year)

movies_embeddings = U[:, :k_best] * s[:k_best]

def show_movie_extremes_for_component(comp, top_n=7):
    comp_values = movies_embeddings[:, comp]
    order = np.argsort(comp_values)

    low_index = order[:top_n]
    high_index = order[-top_n:]

    movie_ids = ratings_matrix.index.to_numpy()

    lows = movies_meta[movies_meta["item_id"].isin(movie_ids[low_index])]
    lows = lows.assign(score=comp_values[low_index]).sort_values("score")

    highs = movies_meta[movies_meta["item_id"].isin(movie_ids[high_index])]
    highs = highs.assign(score=comp_values[high_index]).sort_values("score", ascending=False)

    print(f"\nComponent {comp+1}: Movies at negative end:")
    print(lows[["item_id", "title", "year", "avg_rating", "score"]].to_string(index=False))

    print(f"\nComponent {comp+1}: Movies at positive end:")
    print(highs[["item_id", "title", "year", "avg_rating", "score"]].to_string(index=False))

for comp in range(min(3, movies_embeddings.shape[1])):
    show_movie_extremes_for_component(comp)

# Users
user_embeddings = Vt[:k_best, :].T * s[:k_best]  

def show_user_extremes_for_component(comp, top_n=7):
    comp_values = user_embeddings[:, comp]
    order = np.argsort(comp_values)

    low_idx = order[:top_n]
    high_idx = order[-top_n:]

    users_index = ratings_matrix.columns.to_numpy()

    lows = pd.DataFrame({"user_id": users_index[low_idx], "score": comp_values[low_idx]
    }).sort_values("score")

    highs = pd.DataFrame({"user_id": users_index[high_idx], "score": comp_values[high_idx]
    }).sort_values("score", ascending=False)

    print(f"\nComponent {comp+1}: Users at negative end:")
    print(lows.to_string(index=False))

    print(f"\nComponent {comp+1}: Users at positive end:")
    print(highs.to_string(index=False))

for comp in range(min(3, user_embeddings.shape[1])):
    show_user_extremes_for_component(comp)

"""
The movies in one component are similar in genre and theme. For example for 
component 1 at the negative end, these movies are united by their status as
iconic, high-energy blockbusters from the 1980s and 1990s, blending action,
adventure, and often sci-fi or fantasy elements, with strong audience appeal
and cultural significance. 

Similary, the users in one component like movies with similar genres and themes.
"""

# Pairwise comparision
movies_2d = movies_embeddings[:, :2]
users_2d = user_embeddings[:, :2]

plt.figure(figsize=(8, 6))
plt.scatter(movies_2d[:, 0], movies_2d[:, 1], 
            alpha=0.4, label="Movies", s=10, c="blue")
plt.scatter(users_2d[:, 0], users_2d[:, 1], 
            alpha=0.4, label="Users", s=10, c="red")
plt.axhline(0, color="gray", linewidth=0.5)
plt.axvline(0, color="gray", linewidth=0.5)
plt.xlabel("Latent Factor 1")
plt.ylabel("Latent Factor 2")
plt.title("Users and Movies in Latent Space")
plt.legend()
plt.savefig("Users_and_Movies_in_Latent_Space")
plt.show()

# Example for a user
def plot_user_with_nearest_movies(user_id, top_n=5):
    user_index = list(ratings_matrix.columns).index(user_id)
    user_vec = user_embeddings[user_index, :2]
    
    dists = cosine_distances(
        user_embeddings[user_index, :].reshape(1, -1), 
        movies_embeddings[:, :k_best]
    ).flatten()
    
    nearest_idx = np.argsort(dists)[:top_n]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(movies_2d[:, 0], movies_2d[:, 1], 
                alpha=0.3, label="Movies", s=10, c="blue")
    plt.scatter(users_2d[:, 0], users_2d[:, 1], 
                alpha=0.1, label="Users", s=10, c="red")
    plt.scatter(user_vec[0], user_vec[1], c="black", s=80, marker="x", label=f"User {user_id}")
    plt.scatter(movies_2d[nearest_idx, 0], movies_2d[nearest_idx, 1],
                c="green", s=50, label="Nearest Movies") 
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.xlabel("Latent Factor 1")
    plt.ylabel("Latent Factor 2")
    plt.title(f"User {user_id} and their nearest movies in latent space")
    plt.legend()
    plt.savefig("Users_and_their_nearest_movies")
    plt.show()
    
    movie_ids = ratings_matrix.index.to_numpy()
    print(f"\nNearest {top_n} movies to user {user_id}:")
    print(movie_ids[nearest_idx])

plot_user_with_nearest_movies(user_id=1)


# Clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
movie_labels = kmeans.fit_predict(movies_embeddings)

labels_df = pd.DataFrame({
    "item_id": ratings_matrix.index.to_numpy(),
    "cluster": movie_labels
})
movies_meta = movies_meta.merge(labels_df, on="item_id", how="left")

coords_2d = pca_transform(movies_embeddings, n_components=2)

plt.figure(figsize=(8, 6))
plt.scatter(coords_2d[:, 0], coords_2d[:, 1],
            c=movie_labels, alpha=0.7, s=10)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Movie clusters in latent space")
plt.savefig("Movie_clusters")
plt.show()

print("\nSample movies per cluster:")
for c in range(num_clusters):
    subset = movies_meta[movies_meta["cluster"] == c]
    if subset.empty:
        print(f"\nCluster {c+1}: (no items)")
        continue
    sample = subset.nlargest(min(8, len(subset)), "avg_rating")
    print(f"\nCluster {c+1} (n={len(subset)}):")
    print(sample[["title", "avg_rating", "year"]].to_string(index=False))

"""
Movies in one cluster are similar in theme and audience. 
For example in the frist cluster they emphasize human 
stories, often with emotional or cultural resonance, and 
tend to have limited mainstream commercial success, while
the films in the fifth cluster are iconic 1970s-1980s blockbusters,
primarily in the action-adventure and sci-fi/fantasy genres.
"""
