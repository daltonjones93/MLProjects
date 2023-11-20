import numpy as np

def kmeans(X, k, max_iters=100, tol=1e-4):
    """
    K-means clustering algorithm.

    Parameters:
    - X: numpy array, shape (n_samples, n_features)
      Input data points.
    - k: int
      Number of clusters.
    - max_iters: int, optional, default: 100
      Maximum number of iterations.
    - tol: float, optional, default: 1e-4
      Tolerance to declare convergence.

    Returns:
    - centroids: numpy array, shape (k, n_features)
      Final cluster centers.
    - labels: numpy array, shape (n_samples,)
      Index of the cluster each sample belongs to.
    """

    n_samples, n_features = X.shape

    # Initialize centroids randomly
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

        # Update centroids
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return centroids, labels

# Example usage
# Generate random data for demonstration purposes
np.random.seed(42)
data = np.random.rand(100, 2)

# Set the number of clusters
k = 3

# Apply k-means clustering
centroids, labels = kmeans(data, k)

# Print results
print("Final Centroids:\n", centroids)
print("Labels:\n", labels)
