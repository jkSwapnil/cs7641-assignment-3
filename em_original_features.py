# EM clustering on the Rice and Digits dataset using the original features

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from config import RANDOM_STATE


def execute_em_clustering(X, max_k, random_state=RANDOM_STATE):
    """ Execute EM clustering on the given data over the range of k
    Paramters:
        - X: Input data as numpy array (np.ndarray)
        - max_k: Max. value of the k to explore (int)
        - random_state: Random state to make result replicable (int)
    Returns:
        - Values of k (Number of clusters) used
        - AIC for different k
        - Silhouette scores for different k
    """
    k_values = []  # Different values of k
    aic_scores = []  # List of AIC score for different values of k
    sil_scores = []  # List of silhouette scores for different k values
    for k in tqdm(range(2, max_k+1)):
        em = GaussianMixture(n_components=k, random_state=random_state)
        em.fit(X)
        k_values.append(k)
        aic_scores.append(em.aic(X))
        sil_scores.append(silhouette_score(X=X, labels=em.predict(X)))
    return k_values, aic_scores, sil_scores


if __name__ == "__main__":


    import matplotlib.pyplot as plt

    from config import RANDOM_STATE, RANDOM_STATE_2
    from datasets import RiceData, DigitsData

    # Max number of clustering
    MAX_K = 15

    # Load the Rice data and cluster it using K-Means
    print("\nEM clustering on raw features of the Rice dataset")
    print("- - - - - - - - - - - - - - - - - - - - - - - - -")
    rice_data = RiceData()
    X, _ = rice_data.get_total()
    k_values, aic_scores, sil_scores = execute_em_clustering(X, max_k=MAX_K, random_state=RANDOM_STATE)  # Using RANDOM_STATE
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(1,1,1)
    ax2 = ax1.twinx()
    ax1.plot(k_values, aic_scores, color="blue", marker='o', label=f"AIC (Seed = {RANDOM_STATE})")
    ax2.plot(k_values, sil_scores, color="red", marker='o', label=f"Silhouette score (Seed: {RANDOM_STATE})")
    k_values, aic_scores, sil_scores = execute_em_clustering(X, max_k=MAX_K, random_state=RANDOM_STATE_2)  # Using RANDOM_STATE_2
    ax1.plot(k_values, aic_scores, color="blue", linestyle="--", marker='o', label=f"AIC (Seed = {RANDOM_STATE_2})")
    ax2.plot(k_values, sil_scores, color="red", linestyle="--", marker='o', label=f"Silhouette score (Seed: {RANDOM_STATE_2})")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Akike Information Criterion")
    ax2.set_ylabel("Silhouette score")
    ax1.legend()
    ax2.legend(loc="center right")
    ax1.set_title("AIC and Silhouette score VS k for Rice dataset")
    ax1.set_xticks([_ for _ in range(2, MAX_K+1)])
    fig.savefig("./plots/em_original_features_rice.png")
    plt.close(fig)
    print("- Plot of AIC & Silhouette score VS k for Rice dataset is saved at: './plots/em_original_features_rice.png'")

    # Load the Digits data and cluster it using K-Means
    print("\nEM clustering on raw features of the Digits dataset")
    print("- - - - - - - - - - - - - - - - - - - - - - - - - -")
    digits_data = DigitsData()
    X, _ = digits_data.get_total()
    k_values, aic_scores, sil_scores = execute_em_clustering(X, max_k=MAX_K, random_state=RANDOM_STATE)  # Using RANDOM_STATE
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(1,1,1)
    ax2 = ax1.twinx()
    ax1.plot(k_values, aic_scores, color="blue", marker='o', label=f"AIC (Seed = {RANDOM_STATE})")
    ax2.plot(k_values, sil_scores, color="red", marker='o', label=f"Silhouette score (Seed: {RANDOM_STATE})")
    k_values, aic_scores, sil_scores = execute_em_clustering(X, max_k=MAX_K, random_state=RANDOM_STATE_2)  # Using RANDOM_STATE_2
    ax1.plot(k_values, aic_scores, color="blue", linestyle="--", marker='o', label=f"AIC (Seed = {RANDOM_STATE_2})")
    ax2.plot(k_values, sil_scores, color="red", linestyle="--", marker='o', label=f"Silhouette score (Seed: {RANDOM_STATE_2})")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Akike Information Criterion")
    ax2.set_ylabel("Silhouette score")
    ax1.legend()
    ax2.legend(loc="center right")
    ax1.set_title("AIC and Silhouette score VS k for Digits dataset")
    ax1.set_xticks([_ for _ in range(2, MAX_K+1)])
    fig.savefig("./plots/em_original_features_digits.png")
    plt.close(fig)
    print("- Plot of AIC & Silhouette score VS k for Digits dataset is saved at: './plots/em_original_features_digits.png'")
    print()
