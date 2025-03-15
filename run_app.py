#!/usr/bin/env python
"""
Run script for Document Clustering Application
This handles any initialization steps needed before launching the UI
"""

import os
import sys
import tkinter as tk
import platform

# Handle NLTK data initialization
try:
    # Try to import the simplified TextProcessor first
    from TextProcessor import TextProcessor
except ImportError:
    try:
        import nltk
        # Download required NLTK data if not already available
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("NLTK resources downloaded successfully")
    except Exception as e:
        print(f"Note: Could not initialize NLTK resources: {e}")
        print("Application will use simplified text processing methods")

def main():
    """Main function to run the document clustering application"""
    # Import the main application
    try:
        from doc_cluster_ui import DocumentClusterApp
        
        # Create and run the Tkinter application
        root = tk.Tk()
        app = DocumentClusterApp(root)
        
        # Set proper DPI awareness for Windows
        if platform.system() == "Windows":
            try:
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
            except:
                pass
                
        # Center the window on screen
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = 1000
        window_height = 700
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Run the application
        print("Starting Document Clustering Tool...")
        root.mainloop()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()