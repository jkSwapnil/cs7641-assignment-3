# Implement ICA on Rice dataset
# Perform Kmeans and EM clustering on the reduced dataset

from sklearn.decomposition import FastICA
from scipy.stats import kurtosis

from config import RANDOM_STATE


def execute_ica(X, n_comp):
    """ Execute ICA on the passed data
    Paramters:
        - X: Input data as numpy array (np.ndarray)
        - n_comp: Number of the components to run ICA for (int)
    Returns:
        - Tranformed data with the new features
        - Mean kurtosis of the transformed features
    """
    ica = FastICA(n_components=n_comp, random_state=RANDOM_STATE)
    # Fit PCA model and transform the input data
    _X = ica.fit_transform(X)
    # Evaluate the mean kurtosis of the transformed features
    mean_kurtosis = 0
    for _ in kurtosis(_X, axis=0):
        mean_kurtosis += abs(_)
    mean_kurtosis = mean_kurtosis/n_comp
    return _X, mean_kurtosis


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from config import RANDOM_STATE
    from datasets import RiceData
    from kmeans_original_features import execute_k_means_clustering
    from em_original_features import execute_em_clustering

    # Load the Rice dataset, execute ICA, and KMeans clustering on the transformed data
    print("\nICA on Rice Dataset")
    print("- - - - - - - - - - ")
    rice_data = RiceData()
    X, _ = rice_data.get_total()
    # Plot the variation of kurtosis vs number of components
    n_comps = []  # List of all the number of components considered
    kurts = []  # List of kurtosis for all the number of components
    for _n in range(2, X.shape[1]+1):
        n_comps.append(_n)
        _, kurt = execute_ica(X, n_comp=_n)
        kurts.append(kurt)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(n_comps, kurts, color="green", marker='o', label="Kurtosis VS no. of components")
    ax.set_xlabel("No. of components")
    ax.set_ylabel("Kurtosis")
    ax.set_title("Kurtosis VS No. of components for ICA Rice")
    ax.legend()
    fig.savefig("./plots/ica_kurtosis_vs_components_rice.png")
    plt.close(fig)
    print("- Plot of kurtosis VS No. of components for ICA Rice is saved at: './plots/ica_kurtosis_vs_components_rice.png'")
    # Perform transformation using number of components
    # Using 6 as the optimum number of components
    _X, _ = execute_ica(X, n_comp=6)
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
    ax1.set_title("WCSS and Silhouette score VS k for Rice dataset reduced by ICA")
    fig.savefig("./plots/kmeans_ica_features_rice.png")
    plt.close(fig)
    print("- KMeans plot of WCSS & Silhouette score VS k for ICA Rice is saved at: './plots/kmeans_ica_features_rice.png'")
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
    fig.savefig("./plots/em_ica_features_rice.png")
    plt.close(fig)
    print("- EM plot of AIC & Silhouette score VS k for ICA Rice dataset is saved at: './plots/em_ica_features_rice.png'")
    print()
