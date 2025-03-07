import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

class ClusterMaker:
    def __init__(self, embeddings):
        """
        Initialize the clustering class with a list of embeddings.

        :param embeddings: List of embedding vectors (numpy arrays)
        """
        self.embeddings = np.array(embeddings)
        self.cluster_labels = None

    def cluster_embeddings(self, num_clusters=3):
        """
        Cluster the embeddings using Agglomerative Clustering with cosine similarity.

        :param num_clusters: Number of clusters
        :return: List of cluster labels
        """
        similarity_matrix = cosine_similarity(self.embeddings)  # Compute cosine similarity matrix

        # Perform Agglomerative Clustering with precomputed cosine similarity
        clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity="precomputed", linkage="average")
        self.cluster_labels = clustering.fit_predict(1 - similarity_matrix)  # Convert similarity to distance
        
        return self.cluster_labels

    def plot_embeddings(self):
        """
        Reduce the dimensionality of embeddings using t-SNE and plot them in 2D.
        """
        if self.cluster_labels is None:
            raise ValueError("Run cluster_embeddings() before plotting.")

        # Reduce dimensions to 2D using t-SNE
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        reduced_embeddings = tsne.fit_transform(self.embeddings)

        # Plot the clusters
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=self.cluster_labels, cmap="viridis", alpha=0.7)
        plt.colorbar(scatter, label="Cluster Labels")
        plt.title("t-SNE Visualization of Clustered Embeddings")
        plt.xlabel("TSNE Dimension 1")
        plt.ylabel("TSNE Dimension 2")
        plt.show()
