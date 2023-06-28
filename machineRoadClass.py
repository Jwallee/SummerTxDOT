import os
import numpy as np
import pandas as pd
import warnings
from openpyxl import Workbook
# Filter out the specific warning
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl.styles.stylesheet")

'''
    ROAD CLASS PREDICTION
Data Collected From These Parameters:

Data Used: Roadway System, Outside City Limits, Toll Road/Toll Lane, Narrative
Crash Range: 2022 Austin, TX

Reports downloaded top to bottom based on crash ID
'''

# Specs = folders you are accessing with data
# TOLL BRIDGES DATA NOT FOUND YET


# region WHAT NEEDS TO BE CHANGED
Field = 'Road Class'
specs = ["CityStreet","CountyRoad","FarmToMarket","Interstate","NonTrafficway","OtherRoads","Tollway","US&StateHighways"]
classifiers = ["CITY STREET", "COUNTY ROAD", "FARM TO MARKET", "INTERSTATE", "NON-TRAFFICWAY", "OTHER ROADS", "TOLLWAY", "US & STATE HIGHWAYS"]

RoadwaySystem = []
OutsideCL = []
TollRoad = []
info_variables = ["RoadwaySystem", "OutsideCL", "TollRoad"]
info = [RoadwaySystem, OutsideCL, TollRoad]
columns = ['B','C','E']
dataset_limit = 50
# endregion

# region READING EXCEL SPREADSHEET


# Read the Excel spreadsheet



directory_path = 'crashes/'+Field+'/excel'

for file_name in os.listdir(directory_path):
    file_path = os.path.join(directory_path, file_name)
    file_values = []  # New array for each file
    
    for column in columns:
        df = pd.read_excel(file_path, usecols=column, skiprows=2, nrows = dataset_limit)
        values = df.values.flatten().tolist()
        file_values.append(values)  # Append values to the file array
        
    RoadwaySystem.append(file_values[0])  # Append file array to RoadwaySystem
    OutsideCL.append(file_values[1])  # Append file array to OutsideCL
    TollRoad.append(file_values[2])  # Append file array to TollRoad

# endregion

# region READING TEXT INTO PROGRAM

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
    folder_path ='liftedText/'+Field+"/"+spec  # REPLACE THIS IF THE PDFs ARE STORED ELSEWHERE
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


for run in range(0,len(narrative_array)):
    for assign in range(0,len(narrative_array[run])):
        # print(assign)
        classify.append(classifiers[run])
        total.append(narrative_array[run][assign])
# print(len(total))
# print(classify)

# endregion


# region MACHINE LEARNING MODULE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import scipy.sparse as sp
import re
# # # ACTUAL DATA PROCESSING STUFF

def testing(narratives,classifications, info_variables, info_values):
    # print(info_variables)
    print(len(info_values))
    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the narratives into a feature matrix
    X_narratives = vectorizer.fit_transform(narratives)
    # print(vectorizer.vocabulary_)

    # Encode the informational variables
    label_encoders = {}
    encoded_info = []
    label_encoder = LabelEncoder()



    # PROBLEM CHILD
    for info_set in info_values:
        encoded_values = []
        for info in info_set:
            info = np.array(info, ndmin=1)  # Convert to 1D array
            encoded_value = label_encoder.fit_transform(info)
            encoded_values.append(encoded_value)
        encoded_info.append(encoded_values)
    
    # print(len(encoded_info))
    # Combine the narrative features and encoded informational variables
    feature_matrices = [X_narratives] + [sp.vstack(info).T for info in encoded_info]
    X_combined = sp.hstack(feature_matrices)

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
# endregion


# RUNNING THE CODE
print(len(info))
best_model, vectorizer, label_encoder = testing(total,classify,info_variables,info)