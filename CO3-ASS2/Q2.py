import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

data = pd.read_csv("Q2.csv")

X = data.drop("Sample", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("PCA Explained Variance")
print(pca.explained_variance_ratio_)

fa = FactorAnalysis(
    n_components=2,
    random_state=42
)

X_fa = fa.fit_transform(X_scaled)

ica = FastICA(
    n_components=2,
    random_state=42,
    max_iter=2000,
    whiten="unit-variance"
)

X_ica = ica.fit_transform(X_scaled)

gmm_pca = GaussianMixture(
    n_components=2,
    random_state=42
)

gmm_fa = GaussianMixture(
    n_components=2,
    random_state=42
)

gmm_ica = GaussianMixture(
    n_components=2,
    random_state=42
)

labels_pca = gmm_pca.fit_predict(X_pca)
labels_fa = gmm_fa.fit_predict(X_fa)
labels_ica = gmm_ica.fit_predict(X_ica)

score_pca = silhouette_score(X_pca, labels_pca)
score_fa = silhouette_score(X_fa, labels_fa)
score_ica = silhouette_score(X_ica, labels_ica)

print("\nSilhouette Scores")
print("PCA + GMM:", round(score_pca, 3))
print("FA + GMM:", round(score_fa, 3))
print("ICA + GMM:", round(score_ica, 3))

result = pd.DataFrame({
    "Sample": data["Sample"],
    "PCA_Cluster": labels_pca,
    "FA_Cluster": labels_fa,
    "ICA_Cluster": labels_ica
})

print("\nCluster Comparison")
print(result)

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels_pca,
    cmap="viridis",
    s=100
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA + GMM")
plt.colorbar(label="Cluster")
plt.show()

plt.scatter(
    X_fa[:, 0],
    X_fa[:, 1],
    c=labels_fa,
    cmap="viridis",
    s=100
)

plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.title("Factor Analysis + GMM")
plt.colorbar(label="Cluster")
plt.show()

plt.scatter(
    X_ica[:, 0],
    X_ica[:, 1],
    c=labels_ica,
    cmap="viridis",
    s=100
)

plt.xlabel("Independent Component 1")
plt.ylabel("Independent Component 2")
plt.title("ICA + GMM")
plt.colorbar(label="Cluster")
plt.show()

print("\nTirumalaraju Bhanu Pavan Varma - 192425380")
