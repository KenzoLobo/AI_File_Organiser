# Document Clustering Tool

This tool helps you organize and cluster documents based on their semantic similarity. It can automatically group similar documents into folders, making it easier to manage large collections of files.

## Features

- Process multiple document formats (PDF, DOCX, PPTX, TXT, etc.)
- Automatically extract and analyze text content
- Create semantic embeddings of documents
- Cluster similar documents together
- Visualize document clusters
- Organize files into folders based on clustering results
- User-friendly GUI interface
- Command-line interface for automation

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/document-clustering-tool.git
cd document-clustering-tool
```

### 2. Create a Virtual Environment

#### On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, install the dependencies manually:

```bash
pip install numpy pandas scikit-learn matplotlib nltk sentence-transformers PyMuPDF python-docx openpyxl python-pptx PyInstaller
```

### 4. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

### 5. Running the Application

#### GUI Interface:

```bash
python run_app.py
```

#### Command Line Interface:

```bash
python script.py
```

### 6. Building a Standalone Executable

To create a standalone executable that can run without Python installed:

```bash
python build_app.py
```

This will create an executable in the `dist` directory.

## File Structure

- `run_app.py` - Main entry point for the GUI application
- `doc_cluster_ui.py` - The GUI implementation
- `script.py` - Command-line interface script
- `TextProcessor.py` - Handles document text extraction and processing
- `TextEmbedder.py` - Creates document embeddings using transformer models
- `ClusterMaker.py` - Implements document clustering algorithms
- `build_app.py` - Script to build a standalone executable

## Usage Guide

### Using the GUI

1. Launch the application using `python run_app.py`
2. Browse and select a directory containing your documents
3. Choose between automatic clustering or custom folders
4. Process the documents
5. Review the clustering results
6. Optionally rename clusters
7. Apply clustering to organize your files

### Using the Command Line

1. Run `python script.py`
2. Enter the directory path when prompted
3. Choose between automatic clustering or custom folders
4. Follow the on-screen instructions

## Dependency Details

- **sentence-transformers**: Used for generating document embeddings
- **PyMuPDF (fitz)**: For PDF processing
- **python-docx**: For Word document processing
- **openpyxl**: For Excel file processing
- **python-pptx**: For PowerPoint file processing
- **scikit-learn**: For clustering algorithms
- **matplotlib**: For visualization
- **nltk**: For text processing
- **pandas**: For data handling
- **numpy**: For numerical operations
- **tkinter**: For the GUI (included with Python)
- **PyInstaller**: For creating standalone executables

## Troubleshooting

### Common Issues

1. **NLTK Data Missing**:
   If you see errors about missing NLTK data, run:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

2. **PDF Extraction Errors**:
   Ensure you have the latest version of PyMuPDF:
   ```bash
   pip install --upgrade PyMuPDF
   ```

3. **Building Executable Fails**:
   Make sure PyInstaller is installed and try cleaning its cache:
   ```bash
   pip install pyinstaller
   pyinstaller --clean
   ```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
