import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

data = pd.read_csv("Q1.csv")

X = data[["AnnualIncome", "SpendingScore"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
silhouette_scores = []

for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertia.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.plot(range(2, 7), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

print("Silhouette Scores")

for k, score in zip(range(2, 7), silhouette_scores):
    print("K =", k, "Score =", round(score, 3))

optimal_k = range(2, 7)[silhouette_scores.index(max(silhouette_scores))]

print("Optimal Number of Clusters:", optimal_k)

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X_scaled)

data["Cluster"] = labels

print("\nClustered Data")
print(data)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA Explained Variance")
print(pca.explained_variance_ratio_)

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels,
    cmap="viridis",
    s=100
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segmentation using K-Means and PCA")
plt.colorbar(label="Cluster")
plt.show()

print("\nTirumalaraju Bhanu Pavan Varma - 192425380")
