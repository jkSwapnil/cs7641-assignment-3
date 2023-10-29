# KMeans clustering on the Rice and Digits datasets using the original features

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from config import RANDOM_STATE


def execute_k_means_clustering(X, max_k, random_state=RANDOM_STATE):
    """ Execute KMeans clustering on the given data over the range of k
    Paramters:
        - X: Input data as numpy array (np.ndarray)
        - max_k: Max. value of the k to explore (int)
        - random_state: Random state to make result replicable (int)
    Returns:
        - Values of k (Number of clusters) used
        - Within cluster sum of square
        - Silhouette scores
    """
    # List of within cluster distances for different values of k
    # List of silhouette scores for different k values
    k_values = []
    wcss = []
    sil_scores = []
    for k in tqdm(range(2, max_k+1)):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        k_values.append(k)
        wcss.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X=X, labels=kmeans.labels_))
    return k_values, wcss, sil_scores


if __name__ == "__main__":


    import matplotlib.pyplot as plt

    from config import RANDOM_STATE, RANDOM_STATE_2
    from datasets import RiceData, DigitsData

    # Max number of clustering
    MAX_K = 15

    # Load the Rice data and cluster it using K-Means
    print("\nKMeans clustering on raw features of the Rice dataset")
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - -")
    rice_data = RiceData()
    X, _ = rice_data.get_total()
    k_values, wcss, sil_scores = execute_k_means_clustering(X, max_k=MAX_K, random_state=RANDOM_STATE)  # Using RANDOM_STATE
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(1,1,1)
    ax2 = ax1.twinx()
    ax1.plot(k_values, wcss, color="blue", marker='o', label=f"WCSS (Seed = {RANDOM_STATE})")
    ax2.plot(k_values, sil_scores, color="red", marker='o', label=f"Silhouette score (Seed: {RANDOM_STATE})")
    k_values, wcss, sil_scores = execute_k_means_clustering(X, max_k=MAX_K, random_state=RANDOM_STATE_2)  # Using RANDOM_STATE_2
    ax1.plot(k_values, wcss, color="blue", linestyle='--', marker='o', label=f"WCSS (Seed = {RANDOM_STATE_2})")
    ax2.plot(k_values, sil_scores, color="red", linestyle='--', marker='o', label=f"Silhouette score (Seed: {RANDOM_STATE_2})")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Within cluster sum of squares")
    ax2.set_ylabel("Silhouette score")
    ax1.legend()
    ax2.legend(loc="center right")
    ax1.set_title("WCSS and Silhouette score VS k for Rice dataset")
    ax1.set_xticks([_ for _ in range(2, MAX_K+1)])
    fig.savefig("./plots/kmeans_original_features_rice.png")
    plt.close(fig)
    print("- Plot of WCSS & Silhouette score VS k for Rice dataset is saved at: './plots/kmeans_original_features_rice.png'")

    # Load the Digits data and cluster it using K-Means
    print("\nKMeans clustering on raw features of the Digits dataset")
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    digits_data = DigitsData()
    X, _ = digits_data.get_total()
    k_values, wcss, sil_scores = execute_k_means_clustering(X, max_k=MAX_K, random_state=RANDOM_STATE)  # Using RANDOM_STATE
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(1,1,1)
    ax2 = ax1.twinx()
    ax1.plot(k_values, wcss, color="blue", marker='o', label=f"WCSS (Seed = {RANDOM_STATE})")
    ax2.plot(k_values, sil_scores, color="red", marker='o', label=f"Silhouette score (Seed: {RANDOM_STATE})")
    k_values, wcss, sil_scores = execute_k_means_clustering(X, max_k=MAX_K, random_state=RANDOM_STATE_2)  # Using RANDOM_STATE_2
    ax1.plot(k_values, wcss, color="blue", linestyle='--', marker='o', label=f"WCSS (Seed = {RANDOM_STATE_2})")
    ax2.plot(k_values, sil_scores, color="red", linestyle='--', marker='o', label=f"Silhouette score (Seed: {RANDOM_STATE_2})")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Within cluster sum of squares")
    ax2.set_ylabel("Silhouette score")
    ax1.legend()
    ax2.legend(loc="center right")
    ax1.set_title("WCSS and Silhouette score VS k for Digits dataset")
    ax1.set_xticks([_ for _ in range(2, MAX_K+1)])
    fig.savefig("./plots/kmeans_original_features_digits.png")
    plt.close(fig)
    print("- Plot of WCSS & Silhouette score VS k for Digits dataset is saved at: './plots/kmeans_original_features_digits.png'")
    print()
