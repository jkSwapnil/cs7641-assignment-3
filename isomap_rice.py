# Implement Isomap on Rice dataset
# Perform K-Means and EM on the reduced dataset

from sklearn.manifold import Isomap


def execute_isomap(X, n_comp):
    """ Execute Isomap on the passed data
    Paramters:
        - X: Input data as numpy array (np.ndarray)
        - n_comp: Dimensionality of the projection space (int)
    Returns:
        - Tranformed data with the reduced features
        - Reconstruction error
    """
    isomap = Isomap(n_components=n_comp)
    _X = isomap.fit_transform(X)        # Fitting and transforming using random projection
    return _X, isomap.reconstruction_error()


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from config import RANDOM_STATE
    from datasets import RiceData
    from kmeans_original_features import execute_k_means_clustering
    from em_original_features import execute_em_clustering

    # Load the Rice dataset, execute Isomap, and KMeans clustering on the transformed data
    print("\nIsomap on Rice Dataset")
    print("- - - - - - - - - - - ")
    rice_data = RiceData()
    X, _ = rice_data.get_total()
    # Plot the variation of reconstruction error vs number of components
    n_comps = []  # List of the number of components in output
    reconstruction_errors = []  # List of reconstruction error for all the number of components
    for _n in tqdm(range(1, X.shape[1]+1)):
        n_comps.append(_n)
        _, r_error = execute_isomap(X, n_comp=_n)
        reconstruction_errors.append(r_error)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(n_comps, reconstruction_errors, color="green", marker='o', label="Reconstruction error VS no. of components")
    ax.set_xlabel("No. of output components")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title("Reconstruction error VS No. of components for Isomap Rice")
    ax.legend()
    fig.savefig("./plots/isomap_rerror_vs_components_rice.png")
    plt.close(fig)
    print("- Plot of Reconstruction error VS No. of components for Isomap Rice saved at: './plots/isomap_rerror_vs_components_rice.png'")
    # Perform transformation using number of components
    # Using 3 as the optimum number of components
    _X, _ = execute_isomap(X, n_comp=3)
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
    ax1.set_title("WCSS and Silhouette score VS k for Rice dataset reduced by Isomap")
    fig.savefig("./plots/kmeans_isomap_features_rice.png")
    plt.close(fig)
    print("- KMeans plot of WCSS & Silhouette score VS k for Isomap Rice is saved at: './plots/kmeans_isomap_features_rice.png'")
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
    ax1.set_title("AIC and Silhouette score VS k for Rice dataset reduced by Isomap")
    fig.savefig("./plots/em_isomap_features_rice.png")
    plt.close(fig)
    print("- EM plot of AIC & Silhouette score VS k for Isomap Rice dataset is saved at: './plots/em_isomap_features_rice.png'")
    print()
