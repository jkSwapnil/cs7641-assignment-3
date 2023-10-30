# Implement PCA on Digits dataset
# Perform Kmeans and EM clustering on the reduced dataset


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from config import RANDOM_STATE
    from datasets import DigitsData
    from kmeans_original_features import execute_k_means_clustering
    from em_original_features import execute_em_clustering
    from pca_rice import execute_pca

    # Load the Digits data, execute PCA, and execute KMeans clustering on the reduced data
    print("\nPCA on the Digits dataset")
    print("- - - - - - - - - - - - -")
    digits_data = DigitsData()
    X, _ = digits_data.get_total()
    _X, cum_explained_var, pc_cnt = execute_pca(X)   # Execute PCA
    print("- No. of principal components for 0.99 explained variance: ", pc_cnt)
    # Plot the cummulative explained variance vs no. of components
    n_pc = [_ for _ in range(1, _X.shape[1]+1)]
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(n_pc, cum_explained_var, color='green', marker='o', label="Cummulative Explained Variance")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cummulative Explained Variance")
    ax.set_title("Explained variance VS No. of components for Digits dataset")
    ax.legend()
    fig.savefig("./plots/pca_digits_explained_variance.png")
    plt.close(fig)
    print("- Plot of explained variance for Digits dataset saved at: './plots/pca_digits_explained_variance.png'")
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
    ax1.set_title("WCSS and Silhouette score VS k for Digits dataset reduced by PCA")
    fig.savefig("./plots/kmeans_pca_features_digits.png")
    plt.close(fig)
    print("- KMeans plot of WCSS & Silhouette score VS k for PCA Digits is saved at: './plots/kmeans_pca_features_digits.png'")
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
    ax1.set_title("AIC and Silhouette score VS k for Digits dataset reduced by PCA")
    fig.savefig("./plots/em_pca_features_digits.png")
    plt.close(fig)
    print("- EM plot of AIC & Silhouette score VS k for PCA Digits dataset is saved at: './plots/em_pca_features_digits.png'")
    print()
