from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

class KMeansBaseline:
    """
    A class for applying k-means clustering on feature matrices and visualizing results.
    """

    def __init__(self, n_clusters=5, use_pca=True):
        """
        Initializes the KMeansBaseline class.
        
        Args:
            n_clusters: Number of clusters for k-means (default: 5).
            use_pca: Whether to apply PCA for dimensionality reduction (default: True).
        """
        self.n_clusters = n_clusters
        self.use_pca = use_pca

    def cluster(self, feature_matrix):
        """
        Applies k-means clustering to the feature matrix.

        Args:
            feature_matrix: Feature matrix for all images.

        Returns:
            Cluster labels for each image.
        """
        if self.use_pca:
            print("Applying PCA to reduce dimensionality...")
            pca = PCA(n_components=min(50, feature_matrix.shape[1]))  # Reduce to 50 dimensions or less
            feature_matrix = pca.fit_transform(feature_matrix)

        print(f"Performing k-means clustering with {self.n_clusters} clusters...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(feature_matrix)

        return cluster_labels

    def visualize_clusters(self, cluster_labels, dataset, top_k=5):
        """
        Visualizes images from each cluster.

        Args:
            cluster_labels: Cluster labels for all images.
            dataset: Hugging Face dataset object.
            top_k: Number of images to display per cluster (default: 5).
        """
        num_clusters = np.max(cluster_labels) + 1
        for cluster in range(num_clusters):
            print(f"Cluster {cluster + 1}:")
            cluster_indices = np.where(cluster_labels == cluster)[0]

            plt.figure(figsize=(15, 5))
            for i, idx in enumerate(cluster_indices[:top_k]):
                plt.subplot(1, top_k, i + 1)
                plt.imshow(dataset[int(idx)]['image'])
                plt.title(f"Cluster {cluster + 1} - Image {i + 1}")
                plt.axis('off')

            plt.show()