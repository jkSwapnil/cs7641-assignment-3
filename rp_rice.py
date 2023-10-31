# Implement random projection on Rice dataset
# Perform K-Means and EM on the reduced dataset

import numpy as np
from sklearn.random_projection import GaussianRandomProjection

from config import RANDOM_STATE


def execute_rp(X, n_comp, random_state=RANDOM_STATE):
    """ Execute RP on the passed data
    Paramters:
        - X: Input data as numpy array (np.ndarray)
        - n_comp: Dimensionality of the projection space (int)
        - random_state: Random state to use (int)
    Returns:
        - Tranformed data with the reduced features
        - Mean reconstuction error
    """
    rp = GaussianRandomProjection(n_components=n_comp, compute_inverse_components=True, random_state=random_state)
    _X = rp.fit_transform(X)        # Fitting and transforming using random projection
    X_r = rp.inverse_transform(_X)  # Executing the inverse of the Random projection
    r_error = np.sum((X_r - X)**2)/X.shape[0]
    return _X, r_error


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from config import RANDOM_STATE, RANDOM_STATE_2
    from datasets import RiceData
    from kmeans_original_features import execute_k_means_clustering
    from em_original_features import execute_em_clustering

    # Load the Rice dataset, execute RP, and KMeans clustering on the transformed data
    print("\nRandom Projections on Rice Dataset")
    print("- - - - - - - - - - - - - - - - - -")
    rice_data = RiceData()
    X, _ = rice_data.get_total()
    # Plot the variation of reconstruction error vs number of components
    n_comps = []  # List of the number of components in output
    reconstruction_errors = []    # List of reconstruction error for all the number of components using RANDOM_STATE
    reconstruction_errors_2 = []  # List of reconstruction error for all the number of components using RANDOM_STATE_2
    for _n in range(1, X.shape[1]+1):
        n_comps.append(_n)
        # Random state 1
        _, r_error = execute_rp(X, n_comp=_n, random_state=RANDOM_STATE)
        reconstruction_errors.append(r_error)
        # Random state 2
        _, r_error = execute_rp(X, n_comp=_n, random_state=RANDOM_STATE_2)
        reconstruction_errors_2.append(r_error)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(
        n_comps, reconstruction_errors, color="green", marker='o',
        label=f"Reconstruction error VS no. of components (Seed = {RANDOM_STATE})"
        )
    ax.plot(
        n_comps, reconstruction_errors_2, color="black", linestyle="--", marker='o',
        label=f"Reconstruction error VS no. of components (Seed = {RANDOM_STATE_2})"
        )
    ax.set_xlabel("No. of output components")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title("Reconstruction error VS No. of components for RP Rice")
    ax.legend()
    fig.savefig("./plots/rp_rerror_vs_components_rice.png")
    plt.close(fig)
    print("- Plot of Reconstruction error VS No. of components for RP Rice saved at: './plots/rp_rerror_vs_components_rice.png'")
    # Perform transformation using number of components
    # Using 4 as optimum number of compnents
    _X, _ = execute_rp(X, n_comp=4)
    # Execute KMeans clustering on reduced dataset
    MAX_K = 15
    k_values, wcss, sil_scores = execute_k_means_clustering(X=_X, max_k=MAX_K, random_state=RANDOM_STATE)
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(1,1,1)
    ax2 = ax1.twinx()
    ax1.plot(k_values, wcss, color="blue", marker='o', label=f"WCSS (Seed = {RANDOM_STATE})")
    ax2.plot(k_values, sil_scores, color="red", marker='o', label=f"Silhouette score (Seed: {RANDOM_STATE})")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Within cluster sum of squares")
    ax2.set_ylabel("Silhouette score")
    ax1.legend()
    ax2.legend(loc="center right")
    ax1.set_title("WCSS and Silhouette score VS k for Rice dataset reduced by RP")
    fig.savefig("./plots/kmeans_rp_features_rice.png")
    plt.close(fig)
    print("- KMeans plot of WCSS & Silhouette score VS k for RP Rice is saved at: './plots/kmeans_rp_features_rice.png'")
    # Execute EM clustering on reduced dataset
    k_values, aic_score, sil_scores = execute_em_clustering(_X, max_k=MAX_K, random_state=RANDOM_STATE)
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(1,1,1)
    ax2 = ax1.twinx()
    ax1.plot(k_values, aic_score, color="blue", marker='o', label=f"AIC (Seed = {RANDOM_STATE})")
    ax2.plot(k_values, sil_scores, color="red", marker='o', label=f"Silhouette score (Seed: {RANDOM_STATE})")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Akike Information Criterion")
    ax2.set_ylabel("Silhouette score")
    ax1.legend()
    ax2.legend(loc="center right")
    ax1.set_title("AIC and Silhouette score VS k for Rice dataset reduced by RP")
    fig.savefig("./plots/em_rp_features_rice.png")
    plt.close(fig)
    print("- EM plot of AIC & Silhouette score VS k for RP Rice dataset is saved at: './plots/em_rp_features_rice.png'")
    print()
