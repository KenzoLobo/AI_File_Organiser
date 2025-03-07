import os
import time
import numpy as np
import shutil
from TextProcessor import TextProcessor
from TextEmbedder import TextEmbedder
from ClusterMaker import ClusterMaker

def print_files(file_paths):
    """
    Take a list of file paths and prints each one out.
    
    Args:
        file_paths (list): List of file paths to print.
    """
    for f in file_paths:
        print(f)

def print_status(message, total, current):
    """
    Print a status message with a progress indicator.
    
    Args:
        message (str): The status message.
        total (int): Total number of items.
        current (int): Current item number.
    """
    percentage = (current / total) * 100
    print(f"\r{message} {current}/{total} ({percentage:.1f}%)", end="")
    if current == total:
        print()  # New line when complete

def process_documents(directory_name, supported_extensions=None):
    """
    Process all documents in a directory and generate embeddings.
    
    Args:
        directory_name (str): Path to the directory.
        supported_extensions (list, optional): List of supported file extensions.
        
    Returns:
        tuple: (file_paths, embeddings, text_processor, text_embedder)
    """
    if supported_extensions is None:
        supported_extensions = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', 
                              '.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm',
                              '.py', '.js', '.java', '.c', '.cpp', '.h', '.cs', '.php', '.rb', '.go', '.swift']
    
    text_processor = TextProcessor(directory_name)
    text_embedder = TextEmbedder()
    
    # Get all files
    print("Getting all file paths...")
    all_file_paths = text_processor.get_all_file_paths()
    
    # Filter files by extension
    file_paths = []
    for file_path in all_file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in supported_extensions:
            file_paths.append(file_path)
    
    print(f"Found {len(file_paths)} files with supported extensions.")
    
    # Process each file
    embeddings = []
    processed_files = []
    
    print("Processing files...")
    for i, file_path in enumerate(file_paths):
        print_status("Processing file", len(file_paths), i+1)
        
        try:
            # Extract and clean text
            text = text_processor.extract_text_from_file(file_path)
            if not text:
                continue
                
            clean_text = text_processor.clean_text(text)
            if not clean_text:
                continue
            
            # Generate document embedding
            _, _, doc_embedding = text_embedder.process_document(clean_text)
            
            if doc_embedding is not None:
                processed_files.append(file_path)
                embeddings.append(doc_embedding)
        except Exception as e:
            print(f"\nError processing {file_path}: {str(e)}")
    
    print(f"\nSuccessfully processed {len(processed_files)} out of {len(file_paths)} files.")
    return processed_files, embeddings, text_processor, text_embedder

def get_cluster_choice():
    """
    Ask the user to choose between automatic clustering or custom folders.
    
    Returns:
        str: 'auto' or 'custom'
    """
    while True:
        choice = input("Do you want [a]utomatic clustering or [c]ustom folders? (a/c): ").strip().lower()
        if choice in ['a', 'auto', 'automatic']:
            return 'auto'
        elif choice in ['c', 'custom']:
            return 'custom'
        else:
            print("Invalid choice. Please enter 'a' for automatic or 'c' for custom.")

def get_custom_folders(n_clusters):
    """
    Ask the user to provide custom folder names for each cluster.
    
    Args:
        n_clusters (int): Number of clusters.
        
    Returns:
        dict: Mapping from cluster labels to custom folder names.
    """
    print(f"\nYou have {n_clusters} clusters. Please name each one:")
    custom_folders = {}
    
    for i in range(n_clusters):
        name = input(f"Name for cluster {i}: ").strip()
        if not name:
            name = f"cluster_{i}"
        custom_folders[i] = name
    
    return custom_folders

def get_custom_folder_names():
    """
    Ask the user to provide custom folder names.
    
    Returns:
        list: List of custom folder names.
    """
    print("\nEnter custom folder names (one per line). Enter an empty line when done:")
    folder_names = []
    
    while True:
        name = input(f"Folder {len(folder_names) + 1}: ").strip()
        if not name:
            if folder_names:  # If we already have at least one name
                break
            print("Please enter at least one folder name.")
            continue
        folder_names.append(name)
    
    return folder_names

def assign_files_to_folders(file_paths, embeddings, folder_names, text_processor, text_embedder):
    """
    Assign files to custom folders using text similarity.
    
    Args:
        file_paths (list): List of file paths.
        embeddings (list): List of document embeddings.
        folder_names (list): List of folder names.
        text_processor (TextProcessor): Text processor instance.
        text_embedder (TextEmbedder): Text embedder instance.
        
    Returns:
        dict: Mapping from folder names to lists of file paths.
    """
    # Create embeddings for the folder names
    folder_embeddings = [text_embedder.generate_embedding(name) for name in folder_names]
    
    # Assign each file to the most similar folder
    folder_assignments = {name: [] for name in folder_names}
    
    print("\nAssigning files to folders...")
    for i, (file_path, embedding) in enumerate(zip(file_paths, embeddings)):
        # Calculate similarity to each folder
        similarities = [text_embedder.cosine_similarity(embedding, folder_emb) for folder_emb in folder_embeddings]
        
        # Assign to the most similar folder
        most_similar_idx = similarities.index(max(similarities))
        folder_name = folder_names[most_similar_idx]
        folder_assignments[folder_name].append(file_path)
        
        # Print progress
        print_status("Assigning files", len(file_paths), i+1)
    
    print("\n\nFile assignments:")
    for folder, files in folder_assignments.items():
        print(f"{folder}: {len(files)} files")
    
    return folder_assignments

def determine_optimal_clusters(embeddings, max_clusters=10):
    """
    Determine the optimal number of clusters using the elbow method.
    
    Args:
        embeddings (list): List of document embeddings.
        max_clusters (int): Maximum number of clusters to consider.
        
    Returns:
        int: Optimal number of clusters.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as np
    
    # Convert embeddings to numpy array if not already
    embeddings_array = np.array(embeddings)
    
    # Ensure max_clusters is not greater than the number of samples
    max_clusters = min(max_clusters, len(embeddings_array) - 1)
    
    if max_clusters < 2:
        return 2  # Minimum 2 clusters
    
    print("Determining optimal number of clusters...")
    
    # Initialize variables to track best score
    best_n_clusters = 2
    best_score = -1
    
    # Try different numbers of clusters
    for n_clusters in range(2, max_clusters + 1):
        # Create and fit the KMeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_array)
        
        # Calculate silhouette score
        if len(np.unique(labels)) > 1:  # Ensure we have at least 2 clusters
            score = silhouette_score(embeddings_array, labels)
            print(f"  {n_clusters} clusters: silhouette score = {score:.4f}")
            
            # Update best score and clusters if better
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
    
    print(f"Optimal number of clusters: {best_n_clusters} (score: {best_score:.4f})")
    return best_n_clusters

def main():
    """
    Main function to run the document clustering and organization script.
    """
    print("==== Document Clustering and Organization Tool ====\n")
    
    # Get a valid directory from the user
    directory_name = input("Please enter the directory to process: ")
    while not os.path.isdir(directory_name):
        print("Invalid directory name.")
        directory_name = input("Please enter the directory to process: ")
    
    # Process documents and generate embeddings
    file_paths, embeddings, text_processor, text_embedder = process_documents(directory_name)
    
    if not embeddings:
        print("No valid embeddings were generated. Exiting.")
        return
    
    # Ask user for clustering choice
    choice = get_cluster_choice()
    
    if choice == 'custom':
        # Get custom folder names first
        folder_names = get_custom_folder_names()
        
        # Assign files to custom folders based on semantic similarity
        folder_assignments = assign_files_to_folders(file_paths, embeddings, folder_names, text_processor, text_embedder)
        
        # Create output directory for organized files
        output_dir = os.path.join(directory_name, "organized_files_" + time.strftime("%Y%m%d_%H%M%S"))
        
        # Organize files according to custom assignments
        print(f"\nOrganizing files into {output_dir}...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Copy files to respective folders
        for folder_name, files in folder_assignments.items():
            folder_path = os.path.join(output_dir, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                
            for file_path in files:
                file_name = os.path.basename(file_path)
                dest_path = os.path.join(folder_path, file_name)
                shutil.copy2(file_path, dest_path)
        
        # Create a simple visualization of the custom folders
        print("\nCreating visualization...")
        vis_path = os.path.join(output_dir, "folder_visualization.png")
        
        # Create a special visualization for custom folders
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        # Set perplexity to avoid errors with small datasets
        n_samples = len(embeddings)
        perplexity = min(30, max(5, n_samples - 1))
        
        # Convert embeddings to numpy array if not already
        embeddings_array = np.array(embeddings)
        
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings_array)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Generate colors for each folder
        colors = plt.cm.rainbow(np.linspace(0, 1, len(folder_names)))
        
        # Create a mapping of files to their folder index
        file_to_folder = {}
        for folder_idx, (folder_name, files) in enumerate(folder_assignments.items()):
            for file_path in files:
                file_to_folder[file_path] = (folder_idx, folder_name)
        
        # Plot each file with the color of its folder
        for i, file_path in enumerate(file_paths):
            if file_path in file_to_folder:
                folder_idx, folder_name = file_to_folder[file_path]
                ax.scatter(
                    embeddings_2d[i, 0],
                    embeddings_2d[i, 1],
                    c=[colors[folder_idx]],
                    label=folder_name if file_path == folder_assignments[folder_name][0] else "",
                    alpha=0.7
                )
        
        ax.set_title("Document Organization by Custom Folders")
        ax.set_xlabel("t-SNE Dimension 1 (semantic similarity)")
        ax.set_ylabel("t-SNE Dimension 2 (semantic similarity)")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(vis_path)
        
        print(f"\nDone! Files organized into {output_dir}")
        print(f"Visualization saved to {vis_path}")
        
        # Print summary
        print("\nSummary of organized files:")
        for folder, files in folder_assignments.items():
            print(f"{folder}: {len(files)} files")
    else:
        # Determine optimal number of clusters automatically
        n_clusters = determine_optimal_clusters(embeddings)
        
        # Initialize and fit the ClusterMaker
        print(f"\nClustering documents into {n_clusters} groups...")
        cluster_maker = ClusterMaker(n_clusters=n_clusters)
        labels = cluster_maker.fit(embeddings, file_paths)
        
        # Get clusters
        clusters = cluster_maker.get_clusters()
        
        # Print cluster summary
        print("\nCluster summary:")
        for label, files in clusters.items():
            print(f"Cluster {label}: {len(files)} files")
        
        # Auto naming
        print("\nAutomatically naming clusters...")
        custom_folders = cluster_maker.auto_name_clusters(text_processor)
        print("Suggested cluster names:")
        for label, name in custom_folders.items():
            print(f"Cluster {label}: {name}")
            
        # Ask if user wants to modify the automatic names
        modify = input("\nDo you want to modify these names? (y/n): ").strip().lower()
        if modify in ['y', 'yes']:
            custom_folders = get_custom_folders(n_clusters)
            
        # Create output directory for organized files
        output_dir = os.path.join(directory_name, "organized_files_" + time.strftime("%Y%m%d_%H%M%S"))
        
        # Organize files
        print(f"\nOrganizing files into {output_dir}...")
        organized_files = cluster_maker.organize_files(output_dir, custom_folders)
        
        # Visualize clusters
        print("\nCreating visualization...")
        vis_path = os.path.join(output_dir, "cluster_visualization.png")
        cluster_maker.visualize_clusters(vis_path, custom_folders)
        
        print(f"\nDone! Files organized into {output_dir}")
        print(f"Visualization saved to {vis_path}")
        
        # Print summary
        print("\nSummary of organized files:")
        for folder, files in organized_files.items():
            print(f"{folder}: {len(files)} files")
    
    print("\nThank you for using the Document Clustering and Organization Tool!")

# Run the script
if __name__ == "__main__":
    main()