import os
import numpy as np
import pandas as pd

'''
    CRASH SEVERITY PREDICTION
Data Collected From These Parameters:

Data Used: Narrative, Speed Limit, Weather, Light Condition
Crash Range: 
    Non-Fatal: December 2022 Austin, TX
    Fatal: 2022 Austin, TX

Reports downloaded top to bottom based on crash ID
'''

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
# print(classify)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import scipy.sparse as sp
# # ACTUAL DATA PROCESSING STUFF
def testing(narratives,classifications,speed,light):
    # from sklearn import svm

    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the narratives into a feature matrix
    X_narratives = vectorizer.fit_transform(narratives)
    # print(vectorizer.vocabulary_)


    label_encoder = LabelEncoder()
    speeds_encoded = label_encoder.fit_transform(speed)
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

    grid_search = GridSearchCV(classifier, param_grid, cv=6)
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Make predictions on the testing set
    y_pred = best_model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    # Print narratives, predicted labels, and actual labels
    for narrative, pred_label, actual_label in zip(narratives, y_pred, y_test):
        print("Narrative:", narrative)
        print("Predicted Label:", pred_label)
        print("Actual Label:", actual_label)
        print()
    
    # Return the best model
    return best_model, vectorizer, label_encoder
# print(classify)
# print(total)

best_model, vectorizer, label_encoder = testing(total,classify,speeds,lights)