import os
import time
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import shutil
from collections import defaultdict

# Import your modules
from TextProcessor import TextProcessor
from TextEmbedder import TextEmbedder
from ClusterMaker import ClusterMaker

class DocumentClusterApp:
    def __init__(self, root):
        """Initialize the main application window."""
        self.root = root
        root.title("Document Clustering Tool")
        root.geometry("1000x700")
        root.minsize(800, 600)
        
        # Application state variables
        self.directory_path = tk.StringVar()
        self.status_text = tk.StringVar(value="Ready to start")
        self.progress_var = tk.DoubleVar(value=0)
        self.clustering_method = tk.StringVar(value="auto")
        self.optimal_clusters = tk.IntVar(value=5)
        self.current_clusters = tk.IntVar(value=5)
        self.max_clusters = tk.IntVar(value=10)
        
        # Data storage
        self.file_paths = []
        self.embeddings = []
        self.text_processor = None
        self.text_embedder = None
        self.cluster_maker = None
        self.clusters = {}
        self.custom_folders = {}
        self.selected_folder_names = []
        
        # Setup UI components
        self._create_widgets()
        self._setup_layout()
        
    def _create_widgets(self):
        """Create all the UI widgets."""
        # Directory selection
        dir_frame = ttk.LabelFrame(self.root, text="Directory Selection")
        ttk.Label(dir_frame, text="Document Directory:").pack(side="left", padx=5, pady=5)
        ttk.Entry(dir_frame, textvariable=self.directory_path, width=50).pack(side="left", padx=5, pady=5, fill="x", expand=True)
        ttk.Button(dir_frame, text="Browse", command=self._browse_directory).pack(side="left", padx=5, pady=5)
        self.dir_frame = dir_frame
        
        # Clustering options
        options_frame = ttk.LabelFrame(self.root, text="Clustering Options")
        
        # Method selection
        method_frame = ttk.Frame(options_frame)
        ttk.Label(method_frame, text="Clustering Method:").pack(side="left", padx=5, pady=5)
        ttk.Radiobutton(method_frame, text="Automatic", variable=self.clustering_method, value="auto", 
                        command=self._toggle_clustering_options).pack(side="left", padx=5, pady=5)
        ttk.Radiobutton(method_frame, text="Custom Folders", variable=self.clustering_method, value="custom", 
                        command=self._toggle_clustering_options).pack(side="left", padx=5, pady=5)
        method_frame.pack(fill="x", padx=5, pady=5)
        
        # Auto clustering options
        self.auto_frame = ttk.Frame(options_frame)
        ttk.Label(self.auto_frame, text="Max Clusters:").pack(side="left", padx=5, pady=5)
        ttk.Spinbox(self.auto_frame, from_=2, to=20, textvariable=self.max_clusters, width=5).pack(side="left", padx=5, pady=5)
        ttk.Label(self.auto_frame, text="Optimal Clusters:").pack(side="left", padx=5, pady=5)
        self.optimal_clusters_label = ttk.Label(self.auto_frame, textvariable=self.optimal_clusters)
        self.optimal_clusters_label.pack(side="left", padx=5, pady=5)
        ttk.Button(self.auto_frame, text="Find Optimal", command=self._find_optimal_clusters).pack(side="left", padx=5, pady=5)
        self.auto_frame.pack(fill="x", padx=5, pady=5)
        
        # Custom folders options
        self.custom_frame = ttk.Frame(options_frame)
        ttk.Label(self.custom_frame, text="Custom Folder Names:").pack(side="top", anchor="w", padx=5, pady=5)
        self.custom_listbox = tk.Listbox(self.custom_frame, height=5, selectmode=tk.SINGLE)
        self.custom_listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        folder_buttons_frame = ttk.Frame(self.custom_frame)
        ttk.Button(folder_buttons_frame, text="Add", command=self._add_custom_folder).pack(fill="x", padx=5, pady=2)
        ttk.Button(folder_buttons_frame, text="Edit", command=self._edit_custom_folder).pack(fill="x", padx=5, pady=2)
        ttk.Button(folder_buttons_frame, text="Remove", command=self._remove_custom_folder).pack(fill="x", padx=5, pady=2)
        folder_buttons_frame.pack(side="left", padx=5, pady=5)
        
        # Hide custom options initially
        self.custom_frame.pack_forget()
        
        self.options_frame = options_frame
        
        # Process and results area
        self.notebook = ttk.Notebook(self.root)
        
        # Processing tab
        process_frame = ttk.Frame(self.notebook)
        ttk.Button(process_frame, text="Process Documents", command=self._start_processing).pack(padx=10, pady=10)
        ttk.Separator(process_frame, orient="horizontal").pack(fill="x", padx=10, pady=5)
        ttk.Label(process_frame, text="Progress:").pack(anchor="w", padx=10, pady=5)
        ttk.Progressbar(process_frame, variable=self.progress_var, maximum=100, length=400).pack(fill="x", padx=10, pady=5)
        ttk.Label(process_frame, textvariable=self.status_text).pack(anchor="w", padx=10, pady=5)
        
        # Log area
        ttk.Label(process_frame, text="Processing Log:").pack(anchor="w", padx=10, pady=5)
        self.log_text = tk.Text(process_frame, height=10, width=80, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.notebook.add(process_frame, text="Process")
        
        # Results tab (will be populated later)
        self.results_frame = ttk.Frame(self.notebook)
        ttk.Label(self.results_frame, text="No results yet. Process documents first.").pack(padx=20, pady=20)
        self.notebook.add(self.results_frame, text="Results")
        
        # Control buttons at bottom
        self.bottom_frame = ttk.Frame(self.root)
        ttk.Button(self.bottom_frame, text="Exit", command=self.root.destroy).pack(side="right", padx=10, pady=5)
        
    def _setup_layout(self):
        """Arrange all the widgets in the layout."""
        self.dir_frame.pack(fill="x", padx=10, pady=10)
        self.options_frame.pack(fill="x", padx=10, pady=10)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self.bottom_frame.pack(fill="x", padx=10, pady=10)
        
    def _toggle_clustering_options(self):
        """Show/hide options based on selected clustering method."""
        method = self.clustering_method.get()
        if method == "auto":
            self.custom_frame.pack_forget()
            self.auto_frame.pack(fill="x", padx=5, pady=5)
        else:
            self.auto_frame.pack_forget()
            self.custom_frame.pack(fill="x", padx=5, pady=5)
    
    def _browse_directory(self):
        """Open directory browser dialog."""
        dir_path = filedialog.askdirectory(title="Select Document Directory")
        if dir_path:
            self.directory_path.set(dir_path)
            
    def _add_custom_folder(self):
        """Add a new custom folder name."""
        folder_name = tk.simpledialog.askstring("Add Folder", "Enter folder name:")
        if folder_name and folder_name.strip():
            self.custom_listbox.insert(tk.END, folder_name.strip())
            self._update_selected_folders()
            
    def _edit_custom_folder(self):
        """Edit the selected custom folder name."""
        selected = self.custom_listbox.curselection()
        if selected:
            idx = selected[0]
            current_name = self.custom_listbox.get(idx)
            new_name = tk.simpledialog.askstring("Edit Folder", "Enter new name:", initialvalue=current_name)
            if new_name and new_name.strip():
                self.custom_listbox.delete(idx)
                self.custom_listbox.insert(idx, new_name.strip())
                self._update_selected_folders()
                
    def _remove_custom_folder(self):
        """Remove the selected custom folder name."""
        selected = self.custom_listbox.curselection()
        if selected:
            self.custom_listbox.delete(selected[0])
            self._update_selected_folders()
            
    def _update_selected_folders(self):
        """Update the list of selected folder names."""
        self.selected_folder_names = list(self.custom_listbox.get(0, tk.END))
        
    def _log(self, message):
        """Add a message to the log area."""
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")
        self.root.update_idletasks()
        
    def _update_status(self, message, progress=None):
        """Update status text and progress bar."""
        self.status_text.set(message)
        if progress is not None:
            self.progress_var.set(progress)
        self.root.update_idletasks()
        
    def _start_processing(self):
        """Start document processing in a separate thread."""
        directory = self.directory_path.get()
        if not directory or not os.path.isdir(directory):
            messagebox.showerror("Error", "Please select a valid directory.")
            return
            
        # Disable processing button while running
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Button) and widget["text"] == "Process Documents":
                widget.configure(state="disabled")
                
        # Reset progress
        self.progress_var.set(0)
        self._update_status("Starting document processing...")
        self._log("Starting document processing...")
        
        # Start processing in a separate thread to avoid UI freeze
        threading.Thread(target=self._process_documents, daemon=True).start()
        
    def _process_documents(self):
        """Process documents and generate embeddings."""
        try:
            directory = self.directory_path.get()
            
            supported_extensions = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', 
                               '.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm',
                               '.py', '.js', '.java', '.c', '.cpp', '.h', '.cs', '.php', '.rb', '.go', '.swift']
            
            self._update_status("Initializing text processor and embedder...", 5)
            self.text_processor = TextProcessor(directory)
            self.text_embedder = TextEmbedder()
            
            # Get all files
            self._update_status("Getting file paths...", 10)
            self._log("Getting all file paths...")
            all_file_paths = self.text_processor.get_all_file_paths()
            
            # Filter files by extension
            self.file_paths = []
            for file_path in all_file_paths:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in supported_extensions:
                    self.file_paths.append(file_path)
            
            self._log(f"Found {len(self.file_paths)} files with supported extensions.")
            
            # Process each file
            self.embeddings = []
            processed_files = []
            
            self._log("Processing files...")
            for i, file_path in enumerate(self.file_paths):
                progress = 10 + (i / len(self.file_paths) * 70)  # Scale to 10-80%
                self._update_status(f"Processing file {i+1}/{len(self.file_paths)}", progress)
                
                try:
                    # Extract and clean text
                    text = self.text_processor.extract_text_from_file(file_path)
                    if not text:
                        continue
                        
                    clean_text = self.text_processor.clean_text(text)
                    if not clean_text:
                        continue
                    
                    # Generate document embedding
                    _, _, doc_embedding = self.text_embedder.process_document(clean_text)
                    
                    if doc_embedding is not None:
                        processed_files.append(file_path)
                        self.embeddings.append(doc_embedding)
                except Exception as e:
                    self._log(f"Error processing {os.path.basename(file_path)}: {str(e)}")
            
            self._log(f"Successfully processed {len(processed_files)} out of {len(self.file_paths)} files.")
            
            # Handle clustering method
            if len(processed_files) == 0:
                self._log("No files were successfully processed. Please check the logs for errors.")
                self._update_status("Error: No files processed.", 0)
                self._enable_process_button()
                return
                
            if self.clustering_method.get() == "auto":
                # Find optimal clusters if not already done
                if self.current_clusters.get() != self.optimal_clusters.get():
                    self._find_optimal_clusters_internal()
                
                # Cluster documents
                n_clusters = self.optimal_clusters.get()
                self._update_status(f"Clustering documents into {n_clusters} groups...", 85)
                self._log(f"Clustering documents into {n_clusters} groups...")
                self.cluster_maker = ClusterMaker(n_clusters=n_clusters)
                self.cluster_maker.fit(self.embeddings, processed_files)
                self.clusters = self.cluster_maker.get_clusters()
                
                # Auto name clusters
                self._update_status("Naming clusters...", 90)
                self._log("Automatically naming clusters...")
                self.custom_folders = self.cluster_maker.auto_name_clusters(self.text_processor)
                
                for label, name in self.custom_folders.items():
                    self._log(f"Cluster {label}: {name}")
            else:
                # Custom folders
                if not self.selected_folder_names:
                    self._log("No custom folders specified. Please add folder names.")
                    self._update_status("Error: No custom folders specified.", 0)
                    self._enable_process_button()
                    return
                
                self._update_status("Assigning files to custom folders...", 85)
                self._log("Assigning files to custom folders...")
                self._assign_files_to_custom_folders(processed_files)
            
            self._update_status("Processing complete!", 100)
            self._log("Processing complete!")
            
            # Update UI with results on the main thread
            self.root.after(0, self._update_results_tab)
            
        except Exception as e:
            self._log(f"Error during processing: {str(e)}")
            self._update_status(f"Error: {str(e)}", 0)
            import traceback
            self._log(traceback.format_exc())
        finally:
            self._enable_process_button()
            
    def _enable_process_button(self):
        """Re-enable the process button."""
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Button) and widget["text"] == "Process Documents":
                widget.configure(state="normal")
                
    def _find_optimal_clusters(self):
        """Find optimal number of clusters (button handler)."""
        if not self.embeddings:
            messagebox.showinfo("Info", "Process documents first to determine optimal clusters.")
            return
            
        threading.Thread(target=self._find_optimal_clusters_internal, daemon=True).start()
            
    def _find_optimal_clusters_internal(self):
        """Internal method to find optimal clusters."""
        self._update_status("Finding optimal clusters...", 50)
        self._log("Determining optimal number of clusters...")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(self.embeddings)
        
        # Ensure max_clusters is not greater than the number of samples
        max_clusters = min(self.max_clusters.get(), len(embeddings_array) - 1)
        
        if max_clusters < 2:
            max_clusters = 2  # Minimum 2 clusters
        
        # Initialize variables to track best score
        best_n_clusters = 2
        best_score = -1
        
        # Try different numbers of clusters
        for n_clusters in range(2, max_clusters + 1):
            # Create and fit the KMeans model
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_array)
            
            # Calculate silhouette score
            if len(np.unique(labels)) > 1:  # Ensure we have at least 2 clusters
                score = silhouette_score(embeddings_array, labels)
                self._log(f"  {n_clusters} clusters: silhouette score = {score:.4f}")
                
                # Update best score and clusters if better
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
        
        self._log(f"Optimal number of clusters: {best_n_clusters} (score: {best_score:.4f})")
        
        # Update UI on the main thread
        self.root.after(0, lambda: self.optimal_clusters.set(best_n_clusters))
        self.root.after(0, lambda: self.current_clusters.set(best_n_clusters))
        self.root.after(0, lambda: self._update_status("Optimal clusters determined", None))
        
    def _assign_files_to_custom_folders(self, processed_files):
        """Assign files to custom folders based on semantic similarity."""
        folder_names = self.selected_folder_names
        folder_embeddings = [self.text_embedder.generate_embedding(name) for name in folder_names]
        
        # Initialize cluster dictionary
        self.clusters = {i: [] for i in range(len(folder_names))}
        self.custom_folders = {i: name for i, name in enumerate(folder_names)}
        
        # Assign each file to most similar folder
        for i, (file_path, embedding) in enumerate(zip(processed_files, self.embeddings)):
            # Calculate similarity to each folder
            similarities = [self.text_embedder.cosine_similarity(embedding, folder_emb) for folder_emb in folder_embeddings]
            
            # Assign to most similar folder
            most_similar_idx = similarities.index(max(similarities))
            self.clusters[most_similar_idx].append(file_path)
            
            # Log progress
            if i % 10 == 0 or i == len(processed_files) - 1:
                self._log(f"Assigned {i+1}/{len(processed_files)} files to folders")
                
        # Log results
        for i, folder_name in self.custom_folders.items():
            self._log(f"{folder_name}: {len(self.clusters[i])} files")
    
    def _update_results_tab(self):
        """Update the results tab with clustering information."""
        # Clear existing content
        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        # Create a treeview to display results
        ttk.Label(self.results_frame, text="Clustering Results:", font=('Helvetica', 12, 'bold')).pack(anchor="w", padx=10, pady=5)
        
        # Create frame with scrollbar
        tree_frame = ttk.Frame(self.results_frame)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create treeview
        tree = ttk.Treeview(tree_frame)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)
        
        # Configure columns
        tree["columns"] = ("files")
        tree.column("#0", width=200, minwidth=150)
        tree.column("files", width=100, minwidth=50)
        
        tree.heading("#0", text="Folder/Cluster")
        tree.heading("files", text="File Count")
        
        # Insert data
        for label, files in self.clusters.items():
            folder_name = self.custom_folders.get(label, f"Cluster {label}")
            cluster = tree.insert("", "end", text=folder_name, values=(len(files),))
            
            # Add files to the folder
            for file in files:
                tree.insert(cluster, "end", text=os.path.basename(file), values=("",))
        
        # Add buttons at the bottom
        button_frame = ttk.Frame(self.results_frame)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        # Button to rename clusters (only for auto clustering)
        if self.clustering_method.get() == "auto":
            ttk.Button(button_frame, text="Rename Clusters", command=self._rename_clusters).pack(side="left", padx=5, pady=5)
        
        # Add "Organize Files" button in the results tab
        ttk.Button(button_frame, text="Apply Clustering", command=self._organize_files, 
                  style="Accent.TButton").pack(side="right", padx=5, pady=5)
        
        # Register a custom style for the accent button
        style = ttk.Style()
        style.configure("Accent.TButton", font=('Helvetica', 10, 'bold'))
    
    def _rename_clusters(self):
        """Allow user to rename the auto-generated clusters."""
        if not self.custom_folders:
            messagebox.showinfo("Info", "No clusters available to rename.")
            return
            
        # Create a dialog window
        rename_dialog = tk.Toplevel(self.root)
        rename_dialog.title("Rename Clusters")
        rename_dialog.geometry("400x300")
        rename_dialog.transient(self.root)
        rename_dialog.grab_set()
        
        ttk.Label(rename_dialog, text="Rename clusters:", font=('Helvetica', 11, 'bold')).pack(padx=10, pady=10)
        
        # Create entries for each cluster
        entries = {}
        for label, name in sorted(self.custom_folders.items()):
            frame = ttk.Frame(rename_dialog)
            ttk.Label(frame, text=f"Cluster {label}:").pack(side="left", padx=5, pady=5)
            entry = ttk.Entry(frame, width=30)
            entry.insert(0, name)
            entry.pack(side="left", padx=5, pady=5, fill="x", expand=True)
            frame.pack(fill="x", padx=10, pady=2)
            entries[label] = entry
        
        # Save button
        def save_names():
            for label, entry in entries.items():
                self.custom_folders[label] = entry.get().strip() or f"Cluster {label}"
            rename_dialog.destroy()
            self._update_results_tab()
            
        ttk.Button(rename_dialog, text="Save", command=save_names).pack(pady=10)
            
    def _organize_files(self):
        """Organize files into folders based on clustering results."""
        if not self.clusters:
            messagebox.showinfo("Info", "Process documents first before organizing files.")
            return
            
        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
            
        output_subdir = os.path.join(output_dir, f"organized_files_{time.strftime('%Y%m%d_%H%M%S')}")
        
        # Confirm overwrite if directory exists
        if os.path.exists(output_subdir):
            if not messagebox.askyesno("Confirm", f"Directory {output_subdir} already exists. Overwrite?"):
                return
        
        # Start organizing in a separate thread
        threading.Thread(target=lambda: self._organize_files_thread(output_subdir), daemon=True).start()
            
    def _organize_files_thread(self, output_dir):
        """Thread to organize files into folders."""
        try:
            self._update_status("Organizing files...", 0)
            self._log(f"Organizing files into {output_dir}...")
            
            # Create output directory
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Organize files by copying to respective folders
            total_files = sum(len(files) for files in self.clusters.values())
            files_processed = 0
            
            for label, files in self.clusters.items():
                folder_name = self.custom_folders.get(label, f"Cluster_{label}")
                folder_path = os.path.join(output_dir, folder_name)
                
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    
                for file_path in files:
                    file_name = os.path.basename(file_path)
                    dest_path = os.path.join(folder_path, file_name)
                    
                    shutil.copy2(file_path, dest_path)
                    
                    files_processed += 1
                    progress = (files_processed / total_files) * 100
                    self._update_status(f"Organizing files... ({files_processed}/{total_files})", progress)
                    
            self._update_status("Files organized successfully!", 100)
            self._log(f"Done! Files organized into {output_dir}")
            
            # Show success message
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Files organized successfully into {output_dir}"))
            
        except Exception as e:
            self._log(f"Error organizing files: {str(e)}")
            self._update_status(f"Error: {str(e)}", 0)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to organize files: {str(e)}"))

# Main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentClusterApp(root)
    root.mainloop()