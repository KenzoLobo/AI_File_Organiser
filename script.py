import os
from AI_File_Organiser.TextProcessor import TextProcessor
from AI_File_Organiser.TextEmbedder import TextEmbedder

def print_files(file_paths):
    """Take a list of file paths and prints each one out"""
    for f in file_paths:
        print(f)
    
# Run the script
if __name__ == "__main__":

    #get a valid directory from the user
    directory_name = input("Please enter the directory name: ") # Currently using this test directory /Users/kenzolobo/Documents/CS-175
    while not os.path.isdir(directory_name):
        print("Invalid Directory Name")
        directory_name = input("Please enter the directory name: ")

    text_processor = TextProcessor(directory_name)
    file_paths = text_processor.get_all_file_paths()
    # print_files(file_paths)

    #choosing a pdf file to test the chunking algo
    test_file_path = file_paths[1] #/Users/kenzolobo/Documents/CS-175/retrieve.pdf
    print(test_file_path)

    text = text_processor.extract_text_from_file(test_file_path)
    clean_text = text_processor.clean_text(text)

    #check that text was extracted correctly by printing first 100 characters
    print(clean_text[0:100])

    text_embedder = TextEmbedder()
    #try creating the chunks
    chunks = text_embedder.create_chunks(clean_text)

    #check that chunks are being created properly
    for i in range(0,5):
        print (chunks[i])

    #try creating the embedding
    embedding = text_embedder.generate_embedding(clean_text)

    #check the embedding
    print(embedding)

    #calculate the cosine similarity score between an embedding and itself to check that it works
    similarity = text_embedder.cosine_similarity(embedding, embedding)

    #similarity should be 1
    print("Similarity Score: ", similarity)

    




