import os
from TextProcessor import TextProcessor
from TextEmbedder import TextEmbedder

"""Take a list of file paths and prints each one out"""
def print_files(file_paths):
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
    num_chunks = len(chunks)
    print(f"Total number of chunks: {num_chunks}")

    #check that chunks are being created properly
    for i in range(min(5, num_chunks)):
        print(f"Chunk {i+1}:")
        print(chunks[i])
        print("-" * 50)

    #try creating the embedding

    

