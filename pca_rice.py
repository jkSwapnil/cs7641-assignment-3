# Implement PCA on Rice dataset
# Perform Kmeans and EM clustering on the reduced dataset

from sklearn.decomposition import PCA

from config import RANDOM_STATE


def execute_pca(X):
    """ Execute PCA on the passed data
    Paramters:
        - X: Input data as numpy array (np.ndarray)
    Returns:
        - Tranformed data with the new features
        - Cummulative explained variance
        - Number of principal components needed to explain 99% of the variance
    """
    pca = PCA(random_state=RANDOM_STATE)
    # Fit PCA model and transform the input data
    _X = pca.fit_transform(X)
    # Get the explained variance ratio
    explained_var = pca.explained_variance_ratio_
    # Evaluate the number of principal components to explain 99% of variance
    num_of_principal_dimensions = 0
    total_explained_variance = 0
    for _ in explained_var:
        total_explained_variance += _
        num_of_principal_dimensions += 1
        if total_explained_variance > 0.99:
            break
    # Evaluate the cummlative explained variance
    cum_explained_var = []
    total_explained_variance = 0
    for _ in explained_var:
        total_explained_variance += _
        cum_explained_var.append(total_explained_variance)
    return _X, cum_explained_var, num_of_principal_dimensions


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from config import RANDOM_STATE
    from datasets import RiceData
    from kmeans_original_features import execute_k_means_clustering
    from em_original_features import execute_em_clustering

    # Load the Rice data, execute PCA, and execute KMeans clustering on the reduced data
    print("\nPCA on the Rice dataset")
    print("- - - - - - - - - - - -")
    rice_data = RiceData()
    X, _ = rice_data.get_total()
    _X, cum_explained_var, pc_cnt = execute_pca(X)   # Execute PCA
    print("- No. of principal components for 0.99 explained variance: ", pc_cnt)
    # Plot the cummulative explained variance vs no. of components
    n_pc = [_ for _ in range(1, _X.shape[1]+1)]
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(n_pc, cum_explained_var, color='green', marker='o', label="Cummulative Explained Variance")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cummulative Explained Variance")
    ax.set_title("Explained variance VS No. of components for Rice dataset")
    ax.legend()
    fig.savefig("./plots/pca_rice_explained_variance.png")
    plt.close(fig)
    print("- Plot of explained variance for Rice dataset saved at: './plots/pca_rice_explained_variance.png'")
    # Create the reduced fetures
    _X = _X[:, 0:pc_cnt]
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
    ax1.set_title("WCSS and Silhouette score VS k for Rice dataset reduced by PCA")
    fig.savefig("./plots/kmeans_pca_features_rice.png")
    plt.close(fig)
    print("- KMeans plot of WCSS & Silhouette score VS k for PCA Rice is saved at: './plots/kmeans_pca_features_rice.png'")
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
    ax1.set_title("AIC and Silhouette score VS k for Rice dataset reduced by PCA")
    fig.savefig("./plots/em_pca_features_rice.png")
    plt.close(fig)
    print("- EM plot of AIC & Silhouette score VS k for PCA Rice dataset is saved at: './plots/em_pca_features_rice.png'")
    print()
