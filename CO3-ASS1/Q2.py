import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

digits = load_digits()

X = digits.data
y = digits.target

print("Dataset Shape:", X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(
    n_clusters=10,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

gmm = GaussianMixture(
    n_components=10,
    random_state=42
)

gmm_labels = gmm.fit_predict(X_scaled)

kmeans_ari = adjusted_rand_score(y, kmeans_labels)
gmm_ari = adjusted_rand_score(y, gmm_labels)

kmeans_nmi = normalized_mutual_info_score(y, kmeans_labels)
gmm_nmi = normalized_mutual_info_score(y, gmm_labels)

kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
gmm_silhouette = silhouette_score(X_scaled, gmm_labels)

print("\n----- K-MEANS RESULTS -----")
print("ARI:", round(kmeans_ari, 4))
print("NMI:", round(kmeans_nmi, 4))
print("Silhouette Score:", round(kmeans_silhouette, 4))

print("\n----- GMM / EM RESULTS -----")
print("ARI:", round(gmm_ari, 4))
print("NMI:", round(gmm_nmi, 4))
print("Silhouette Score:", round(gmm_silhouette, 4))


plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    cmap="tab10",
    s=10
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Digits Clustering using K-Means")
plt.colorbar(label="Cluster")

plt.show()

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=gmm_labels,
    cmap="tab10",
    s=10
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Digits Clustering using GMM (EM Algorithm)")
plt.colorbar(label="Cluster")

plt.show()

print("T. Bhanu Pavan Varma - 192425380")
