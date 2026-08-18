import numpy as np

image = np.array([
    [50, 52, 49],
    [51, 255, 50],
    [48, 53, 51]
], dtype=float)

print("Original Image:")
print(image)

updated = image.copy()

for i in range(1, image.shape[0] - 1):
    for j in range(1, image.shape[1] - 1):

        neighbors = [
            image[i - 1, j],
            image[i + 1, j],
            image[i, j - 1],
            image[i, j + 1]
        ]

        updated[i, j] = np.mean(neighbors)

print("\nUpdated Image:")
print(updated)

print("\nOriginal Center Pixel:", image[1, 1])
print("Updated Center Pixel:", updated[1, 1])

print("\nT. Bhanu Pavan Varma - 192425380")
