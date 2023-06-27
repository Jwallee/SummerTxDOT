import os
import numpy as np
import pandas as pd


specs = ["Non","Fatal"]
classifiers = ["N - NOT INJURED", "K - FATAL INJURY"]
speeds = []
weathers = []
lights = []

# LITERALLY JUST READING INFO
# Here, we read the files present in the folder path specified (crashes)
def get_file_names(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    return file_names

# Read the Excel spreadsheet
sdf = pd.read_excel('crashes/NonDec2022Update.xlsx', usecols='B', skiprows=2)
wdf = pd.read_excel('crashes/NonDec2022Update.xlsx', usecols='D', skiprows=2)
ldf = pd.read_excel('crashes/NonDec2022Update.xlsx', usecols='C', skiprows=2)
rawS = sdf.values
for a in rawS:
    for l in a:
        speeds.append(l)
rawW = wdf.values
for a in rawW:
    for l in a:
        weathers.append(l)
rawL = ldf.values
for a in rawL:
    for l in a:
        lights.append(l)

sdf = pd.read_excel('crashes/Fatal2022Update.xlsx', usecols='B', skiprows=2)
wdf = pd.read_excel('crashes/Fatal2022Update.xlsx', usecols='D', skiprows=2)
ldf = pd.read_excel('crashes/Fatal2022Update.xlsx', usecols='C', skiprows=2)
rawS = sdf.values
for a in rawS:
    for l in a:
        speeds.append(l)
rawW = wdf.values
for a in rawW:
    for l in a:
        weathers.append(l)
rawL = ldf.values
for a in rawL:
    for l in a:
        lights.append(l)
# print(speeds)


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
def testing(narratives,classifications,speed,light):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    import scipy.sparse as sp
    # from sklearn import svm

    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the narratives into a feature matrix
    X_narratives = vectorizer.fit_transform(narratives)
    # print(vectorizer.vocabulary_)


    label_encoder = LabelEncoder()
    speeds_encoded = label_encoder.fit_transform(speed)
    # weather_encoded = label_encoder.fit_transform(weather)
    light_encoded = label_encoder.fit_transform(light)

    X_combined = sp.hstack([X_narratives, sp.csr_matrix(speeds_encoded). T, sp.csr_matrix(light_encoded).T])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_combined, classifications, test_size=0.2, random_state=42)
    
    
    param_grid = {
    'n_estimators': [200, 300, 500, 600, 800],
    'max_depth': [25, 50, 75, 100, 125]
    }



    # Instantiate and train the logistic regression classifier
    classifier = RandomForestClassifier()
    # Checking efficienceies of each parameter combinaion

    grid_search = GridSearchCV(classifier, param_grid, cv=8)
    grid_search.fit(X_train, y_train)
    results = grid_search.cv_results_
    params = results['params']
    mean_scores = results['mean_test_score']
    # Print the accuracies for each parameter combination
    for param, score in zip(params, mean_scores):
        print("Parameters:", param)
        print("Mean Accuracy:", score)
        print()



    # classifier.fit(X_train, y_train)

    # Make predictions on the testing set
    # test_case = ["Passenger in unit 1 had a no injuries"]
    # test_case_transformed = vectorizer.transform(test_case)

    # best_params = grid_search.best_params_
    # best_model = grid_search.best_estimator_
    # y_pred = best_model.predict(X_test)
    # y_pred = classifier.predict(X_test)

    # y_pred = classifier.predict(test_case_transformed)
    

    # # Evaluate the model's performance
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)

    # Perform cross-validation
    # cv_scores = cross_val_score(classifier, X_combined, classifications, cv=5)

    # # Print the cross-validation scores
    # print("Cross-Validation Scores:", cv_scores)
    # print("Mean Accuracy:", cv_scores.mean())





    # print(X_test)
    # print(y_test)
    # print(y_pred)
# print(classify)
# print(total)

# testing(total,classify,speeds,weathers,lights)
testing(total,classify,speeds,lights)