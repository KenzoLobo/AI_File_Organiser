from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

class TextEmbedder:
    """
    A class to process text by creating overlapping chunks and generating embeddings.

    I have chose a maximum chunk size supported my the SentenceTransformer Library with an overlap of 10% to ensure continuity while reducing redundancy

    Documentation: https://sbert.net/
    """

    def __init__(self, model=None, chunk_size=512, overlap=51): 
        """
        Initializes the TextEmbedder.

        Args:
            model (SentenceTransformer, optional): The model to use. If None, loads the default model.
            chunk_size (int): Number of words per chunk.
            overlap (int): Number of words to overlap between chunks.
        """
        if model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap

    def create_chunks(self, text):
        """
        Splits the input text into overlapping chunks.
        
        Args:
            text (str): The text to chunk.
            
        Returns:
            list: A list of text chunks.
        """
        words = text.split()
        chunks = []

        if not words:
            return chunks

        i = 0
        while i < len(words):
            end_idx = min(i + self.chunk_size, len(words))
            chunk = ' '.join(words[i:end_idx])
            chunks.append(chunk)
            i += (self.chunk_size - self.overlap)

        return chunks

    def generate_embedding(self, text):
        """
        Generates an embedding for the given text by creating chunks, 
        computing embeddings for each chunk, and averaging them.
        
        Args:
            text (str): The text to embed.
            
        Returns:
            numpy.ndarray: The average embedding vector.
        """
        try:
            text_chunks = self.create_chunks(text)
            # Compute embeddings for each chunk
            embeddings = self.model.encode(text_chunks)
            # Compute the average embedding
            avg_embedding = np.mean(embeddings, axis=0)
            return avg_embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None
            
    def cosine_similarity(self, embedding1, embedding2):
        """
        Calculate the cosine similarity between two embedding vectors.
        
        Args:
            embedding1: First embedding vector (numpy array)
            embedding2: Second embedding vector (numpy array)
            
        Returns:
            float: Cosine similarity score
        """
        embedding1 = np.array(embedding1).reshape(1, -1)  # Ensure proper shape
        embedding2 = np.array(embedding2).reshape(1, -1)
        similarity = sklearn_cosine_similarity(embedding1, embedding2)[0][0]  # Extract single value
        return similarity
    
    def process_document(self, text):
        """
        Process a document by chunking it and generating embeddings for each chunk.
        
        Args:
            text (str): The document text.
            
        Returns:
            tuple: (chunks, embeddings, document_embedding)
                chunks (list): The text chunks.
                embeddings (list): Embeddings for each chunk.
                document_embedding (numpy.ndarray): Aggregated embedding for the entire document.
        """
        # Create chunks
        chunks = self.create_chunks(text)
        
        # Generate embeddings for each chunk
        chunk_embeddings = []
        for chunk in chunks:
            embedding = self.generate_embedding(chunk)
            if embedding is not None:
                chunk_embeddings.append(embedding)
        
        # If no embeddings were generated, return empty results
        if not chunk_embeddings:
            return chunks, [], None
        
        # Create a document-level embedding by averaging chunk embeddings
        document_embedding = np.mean(chunk_embeddings, axis=0)
        
        return chunks, chunk_embeddings, document_embedding


# Example Usage
if __name__ == "__main__":
    text = "This is a sample text to demonstrate chunking into overlapping sections for better text processing and retrieval tasks."
    
    embedder = TextEmbedder(chunk_size=10, overlap=3)
    chunks, embeddings, doc_embedding = embedder.process_document(text)

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        print(f"Chunk {idx+1}: {chunk}")
        print(f"Embedding {idx+1}: {embedding[:5]}... (truncated)")
        
    print(f"Document Embedding: {doc_embedding[:5]}... (truncated)")