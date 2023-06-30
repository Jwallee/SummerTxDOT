import os

def read_files_in_folder(folder_path):
    files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                content = file.read()
                files.append(content)
    return files

def find_string_in_documents(string, documents):
    found_documents = []
    for doc in documents:
        if string in doc:
            found_documents.append(doc)
    return found_documents

folder_path = 'liftedText/Road Class/CountyRoad'  # Specify the path to the folder containing the text files
search_string = "SINGLE VEHICLE COLLISION WITH GUARD RAIL. UPON OBSERVATION"  # Specify the string you want to search for

documents = read_files_in_folder(folder_path)
found_docs = find_string_in_documents(search_string, documents)

print(found_docs)