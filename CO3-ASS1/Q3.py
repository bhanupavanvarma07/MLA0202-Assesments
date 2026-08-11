import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis, FastICA

wine = load_wine()

X = wine.data
y = wine.target

print("Original Dataset Shape:", X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("PCA Total Variance:",
      round(sum(pca.explained_variance_ratio_), 4))

fa = FactorAnalysis(
    n_components=2,
    random_state=42
)

X_fa = fa.fit_transform(X_scaled)

print("\nFactor Analysis Output Shape:")
print(X_fa.shape)

ica = FastICA(
    n_components=2,
    random_state=42,
    max_iter=1000
)

X_ica = ica.fit_transform(X_scaled)

print("\nICA Output Shape:")
print(X_ica.shape)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis",
    s=50
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Wine Dataset - PCA")
plt.colorbar(label="Wine Class")

plt.show()

plt.figure(figsize=(8, 6))

plt.scatter(
    X_fa[:, 0],
    X_fa[:, 1],
    c=y,
    cmap="viridis",
    s=50
)

plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.title("Wine Dataset - Factor Analysis")
plt.colorbar(label="Wine Class")

plt.show()

plt.figure(figsize=(8, 6))

plt.scatter(
    X_ica[:, 0],
    X_ica[:, 1],
    c=y,
    cmap="viridis",
    s=50
)

plt.xlabel("Independent Component 1")
plt.ylabel("Independent Component 2")
plt.title("Wine Dataset - ICA")
plt.colorbar(label="Wine Class")

plt.show()

print("\nT. Bhanu Pavan Varma - 192425380")
