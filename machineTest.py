import os

# LITERALLY JUST READING INFO

# Here, we read the files present in the folder path specified (crashes)
def get_file_names(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    return file_names

narrative_array = []

folder_path ='liftedText'  # REPLACE THIS IF THE PDFs ARE STORED ELSEWHERE
file_names = get_file_names(folder_path)

for name in file_names:
    file_path = "liftedText/"+name
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read the contents of the file line by line
        lines = file.read()
        text_string = lines.replace("\n", " ")
        text_string = text_string.upper()
        narrative_array.append(text_string)

print(len(narrative_array))

# ACTUAL DATA PROCESSING STUFF

# These classifications define what the heck each of the reports are in severe/non-severe.
classifications = ["K - FATAL INJURY","K - FATAL INJURY","K - FATAL INJURY","K - FATAL INJURY","K - FATAL INJURY","K - FATAL INJURY","K - FATAL INJURY","K - FATAL INJURY","K - FATAL INJURY","K - FATAL INJURY","B - MINOR INJURY","B - MINOR INJURY","B - MINOR INJURY","B - MINOR INJURY","B - MINOR INJURY","B - MINOR INJURY","B - MINOR INJURY","B - MINOR INJURY","B - MINOR INJURY","B - MINOR INJURY","N - NOT INJURED","N - NOT INJURED","N - NOT INJURED","N - NOT INJURED","N - NOT INJURED","N - NOT INJURED","N - NOT INJURED","N - NOT INJURED","N - NOT INJURED","N - NOT INJURED"]

def testing(narratives,classifications):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the narratives into a feature matrix
    X = vectorizer.fit_transform(narratives)
    # print(vectorizer.vocabulary_)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, classifications, test_size=0.2, random_state=42)

    # Instantiate and train the logistic regression classifier
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = classifier.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # print(X_test)
    print(y_test)
    print(y_pred)

testing(narrative_array,classifications)