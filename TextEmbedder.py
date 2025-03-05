from sentence_transformers import SentenceTransformer
class TextEmbedder:
    """
    A class to process text by creating overlapping chunks and generating embeddings.

    I have chose a maximum chunk size supported my the SentenceTransformer Library with an overlap of 10% to ensure continuity while reducing redundancy

    Documentation: https://sbert.net/
    """

    def __init__(self, model = SentenceTransformer("all-MiniLM-L6-v2"), chunk_size=512, overlap=51): 
        """
        Initializes the TextEmbedder.

        Args:
            model (str): The model to use.
            chunk_size (int): Number of words per chunk.
            overlap (int): Number of words to overlap between chunks.
        """
        self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap

    def create_chunks(self, text):
        """
        Splits the input text into overlapping chunks.
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
        Generates an embedding for the given text
        """
        return None

# Example Usage
if __name__ == "__main__":
    text = "This is a sample text to demonstrate chunking into overlapping sections for better text processing and retrieval tasks."
    
    embedder = TextEmbedder(chunk_size=10, overlap=3)
    results = embedder.process_text(text)

    for idx, (chunk, embedding) in enumerate(results):
        print(f"Chunk {idx+1}: {chunk}")
        print(f"Embedding {idx+1}: {embedding[:5]}... (truncated)")
