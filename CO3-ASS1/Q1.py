import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

data = pd.DataFrame({
    "Age": [
        19,21,20,23,31,22,35,23,64,30,
        67,35,58,24,37,22,35,20,52,35,
        35,25,46,31,54,29,45,35,40,23
    ],
    "Annual_Income": [
        15,15,16,16,17,17,18,18,19,19,
        20,20,21,21,22,22,23,23,24,24,
        25,25,26,26,27,27,28,28,29,29
    ],
    "Spending_Score": [
        39,81,6,77,40,76,6,94,3,72,
        14,99,15,77,13,79,35,66,29,98,
        35,73,5,73,14,82,32,61,31,87
    ]
})

print("Customer Dataset:")
print(data.head())

X = data[["Age", "Annual_Income", "Spending_Score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []

for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(range(2, 11), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

scores = []

for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append(score)

print("\nSilhouette Scores:")

for k, score in zip(range(2, 11), scores):
    print("K =", k, "Score =", round(score, 3))

best_k = range(2, 11)[scores.index(max(scores))]

print("\nOptimal Number of Clusters:", best_k)

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X_scaled)

data["Cluster"] = labels

print("\nClustered Data:")
print(data)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels,
    cmap="viridis",
    s=60
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segmentation using K-Means and PCA")
plt.colorbar(label="Cluster")

plt.show()

print("T. Bhanu Pavan Varma - 192425380")
