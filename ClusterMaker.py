from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from collections import defaultdict

class ClusterMaker:
    """
    A class to cluster document embeddings and organize files based on the clusters.
    """
    
    def __init__(self, n_clusters=5):
        """
        Initialize the ClusterMaker with the desired number of clusters.
        
        Args:
            n_clusters (int): The number of clusters to create.
        """
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.embeddings = None
        self.file_paths = None
        self.labels = None
        
    def fit(self, embeddings, file_paths):
        """
        Fit the clustering model on the document embeddings.
        
        Args:
            embeddings (list): List of document embeddings.
            file_paths (list): List of file paths corresponding to the embeddings.
        """
        self.embeddings = np.array(embeddings)
        self.file_paths = file_paths
        self.labels = self.model.fit_predict(self.embeddings)
        return self.labels
        
    def get_clusters(self):
        """
        Get the clusters and the files in each cluster.
        
        Returns:
            dict: A dictionary mapping cluster labels to lists of file paths.
        """
        if self.labels is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        clusters = defaultdict(list)
        for file_path, label in zip(self.file_paths, self.labels):
            clusters[label].append(file_path)
            
        return dict(clusters)
    
    def auto_name_clusters(self, text_processor):
        """
        Automatically generate names for clusters based on their content.
        
        Args:
            text_processor (TextProcessor): A text processor to extract content from files.
            
        Returns:
            dict: A dictionary mapping cluster labels to cluster names.
        """
        if self.labels is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        clusters = self.get_clusters()
        cluster_names = {}
        
        for label, files in clusters.items():
            # Get the most common words in each cluster
            all_text = ""
            for file in files[:5]:  # Use only first few files for efficiency
                text = text_processor.extract_text_from_file(file)
                all_text += " " + text
                
            # Clean and analyze text to find representative terms
            clean_text = text_processor.clean_text(all_text)
            words = clean_text.split()
            word_freq = defaultdict(int)
            
            for word in words:
                word_freq[word] += 1
            
            # Get the top words as the cluster name
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
            cluster_names[label] = "_".join([word for word, freq in top_words])
            
        return cluster_names
    
    def organize_files(self, output_dir, custom_folders=None):
        """
        Organize files into folders based on clusters or custom folder names.
        
        Args:
            output_dir (str): The directory where organized files will be placed.
            custom_folders (dict, optional): A dictionary mapping cluster labels to custom folder names.
        
        Returns:
            dict: A dictionary mapping folder names to lists of files.
        """
        if self.labels is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        clusters = self.get_clusters()
        organized_files = {}
        
        for label, files in clusters.items():
            if custom_folders and label in custom_folders:
                folder_name = custom_folders[label]
            else:
                folder_name = f"cluster_{label}"
                
            folder_path = os.path.join(output_dir, folder_name)
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                
            # Copy files to the folder
            folder_files = []
            for file in files:
                file_name = os.path.basename(file)
                dest_path = os.path.join(folder_path, file_name)
                shutil.copy2(file, dest_path)
                folder_files.append(file_name)
                
            organized_files[folder_name] = folder_files
            
        return organized_files
    
    def visualize_clusters(self, output_path=None, custom_labels=None):
        """
        Create a visualization of the clusters using t-SNE for dimensionality reduction.
        
        Args:
            output_path (str, optional): Path to save the visualization. If None, the plot is displayed.
            custom_labels (dict, optional): A dictionary mapping cluster labels to custom names.
            
        Returns:
            tuple: The figure and axes objects from matplotlib.
        """
        if self.embeddings is None or self.labels is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        # Reduce dimensionality for visualization
        # Set perplexity to min(30, n_samples - 1) to avoid the error
        n_samples = len(self.embeddings)
        perplexity = min(30, max(5, n_samples - 1))  # Ensure perplexity is between 5 and n_samples-1
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(self.embeddings)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Generate colors for each cluster
        unique_labels = np.unique(self.labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each cluster
        for label, color in zip(unique_labels, colors):
            mask = self.labels == label
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                label=custom_labels[label] if custom_labels and label in custom_labels else f"Cluster {label}",
                alpha=0.7
            )
            
        ax.set_title("Document Clusters")
        ax.set_xlabel("t-SNE Dimension 1 (semantic similarity)")
        ax.set_ylabel("t-SNE Dimension 2 (semantic similarity)")
        ax.legend()
        
        # Save or display the plot
        if output_path:
            plt.tight_layout()
            plt.savefig(output_path)
        else:
            plt.tight_layout()
            
        return fig, ax
    
    def adjust_clusters(self, n_clusters=None):
        """
        Adjust the number of clusters and refit the model.
        
        Args:
            n_clusters (int, optional): The new number of clusters.
            
        Returns:
            list: The new cluster labels.
        """
        if n_clusters:
            self.n_clusters = n_clusters
            self.model = KMeans(n_clusters=n_clusters, random_state=42)
            
        if self.embeddings is None:
            raise ValueError("No embeddings to cluster. Call fit() with embeddings first.")
            
        self.labels = self.model.fit_predict(self.embeddings)
        return self.labels
    
    def use_dbscan(self, eps=0.5, min_samples=5):
        """
        Use DBSCAN clustering algorithm instead of KMeans.
        
        Args:
            eps (float): The maximum distance between samples for them to be in the same neighborhood.
            min_samples (int): The minimum number of samples in a neighborhood for a point to be a core point.
            
        Returns:
            list: The cluster labels.
        """
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        
        if self.embeddings is None:
            raise ValueError("No embeddings to cluster. Call fit() with embeddings first.")
            
        self.labels = self.model.fit_predict(self.embeddings)
        return self.labels