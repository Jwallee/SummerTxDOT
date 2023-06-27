import os
import numpy as np

specs = ["Non","Fatal"]
# classifiers = ["N - NOT INJURED","B - MINOR INJURY", "K - FATAL INJURY"]
classifiers = ["N - NOT INJURED", "K - FATAL INJURY"]

# LITERALLY JUST READING INFO
# Here, we read the files present in the folder path specified (crashes)
def get_file_names(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    return file_names

# FULL ARRAY
narrative_array = []

for spec in specs:
    # print(spec)
    narrative_line = []
    folder_path ='liftedText/'+spec  # REPLACE THIS IF THE PDFs ARE STORED ELSEWHERE
    file_names = get_file_names(folder_path)

    for name in file_names:
        file_path = folder_path+"/"+name
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read the contents of the file line by line
            lines = file.read()
            text_string = lines.replace("\n", " ")
            text_string = text_string.upper()
            narrative_line.append(text_string)
    # Adding to main matrix
    narrative_array.append(narrative_line)


groups = len(narrative_array)
# print(groups)
values = len(narrative_array[0])
# print(values)
total = []
classify = []
for run in range(0,groups):
    for assign in range(0,len(narrative_array[0])):
        classify.append(classifiers[run])
        total.append(narrative_array[run][assign])
# print(len(total))
# print(len(classify))


# # ACTUAL DATA PROCESSING STUFF
def testing(narratives,classifications):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    # from sklearn import svm

    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the narratives into a feature matrix
    X = vectorizer.fit_transform(narratives)
    # print(vectorizer.vocabulary_)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, classifications, test_size=0.2, random_state=42)
    param_grid = {
    'n_estimators': [50, 100, 150, 200, 250],
    'max_depth': [None, 5, 10, 15, 20]
    }



    # Instantiate and train the logistic regression classifier
    classifier = RandomForestClassifier()
    grid_search = GridSearchCV(classifier, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # classifier.fit(X_train, y_train)

    # Make predictions on the testing set
    # test_case = ["Passenger in unit 1 had a no injuries"]
    # test_case_transformed = vectorizer.transform(test_case)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    # y_pred = classifier.predict(X_test)
    # y_pred = classifier.predict(test_case_transformed)
    

    # # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # print(X_test)
    # print(y_test)
    # print(y_pred)
# print(classify)
# print(total)
testing(total,classify)