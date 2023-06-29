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
specs = np.array(["CityStreet","CountyRoad","FarmToMarket","Interstate","NonTrafficway","OtherRoads","Tollway","US&StateHighways"])
classifiers = np.array(["CITY STREET", "COUNTY ROAD", "FARM TO MARKET", "INTERSTATE", "NON-TRAFFICWAY", "OTHER ROADS", "TOLLWAY", "US & STATE HIGHWAYS"])

RoadwaySystem = []
OutsideCL = []
TollRoad = []
info_variables = ["RoadwaySystem", "OutsideCL", "TollRoad"]
info = np.array([])
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
        df = pd.read_excel(file_path, usecols=column, skiprows=2, nrows=dataset_limit)
        values = df.values.flatten().tolist()
        file_values.append(values)  # Append values to the file array
        
    RoadwaySystem.extend(file_values[0])  # Extend RoadwaySystem array
    OutsideCL.extend(file_values[1])  # Extend OutsideCL array
    TollRoad.extend(file_values[2])  # Extend TollRoad array

# Convert lists to NumPy arrays
RoadwaySystem = np.array(RoadwaySystem)
OutsideCL = np.array(OutsideCL)
TollRoad = np.array(TollRoad)

# print(RoadwaySystem.shape)

# Create info array and reshape it
info = np.array([RoadwaySystem, OutsideCL, TollRoad])
info = info.reshape(3, 341)

# print(info.shape)
# print(info)
# print(info[0].shape)
# endregion

# region READING TEXT INTO PROGRAM

def get_file_names(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    return file_names

# FULL ARRAY
narrative_array = np.array([])
# used to create classifications array
sizer = []

for spec in specs:
    # print(spec)
    narrative_line = np.array([])
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
            narrative_line = np.append(narrative_line, text_string)
    sizer.append(len(narrative_line))

    # Adding to main matrix
    narrative_array = np.append(narrative_array, narrative_line, axis = 0)


groups = len(narrative_array)
# print(groups)
values = len(narrative_array[0])
# print(values)
classify = np.array([])
total = np.array([])

for run in range(len(classifiers)):
    for assign in range(sizer[run]):
        classify = np.append(classify, classifiers[run])
        # total = np.append(total, narrative_array[run][assign])
# print(len(total))
# print(classify)

# endregion





# region MACHINE LEARNING MODULE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import scipy.sparse as sp
import re
# # # ACTUAL DATA PROCESSING STUFF

def testing(narratives,classifications,info_values):
    # Checking sizes
    # print(len(info_values))
    # print(len(info_values[0]))
    # print(len(info_values[1]))
    # print(len(info_values[2]))
    # print(len(classifications))
    # print(len(narratives))

    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the narratives into a feature matrix
    X_narratives = vectorizer.fit_transform(narratives)
    # print(vectorizer.vocabulary_)

    # Prepare the data for MultiLabelBinarizer
    info_types = []
    for road_info in info_values:
        road_types = []
        for type_data in road_info:
            road_types.append(type_data)
        info_types.append(road_types)


    # Encode the informational variables
    # print(info_types)
    # print(len(info_types))
    # print(len(info_types[0]))
    # print(len(encoded_info[0]))

    # Split the encoded_info into separate variables for each road
    # Encode the informational variables for each road
    encoded_info_roads = []
    for road_info in info_types:
        print(len(road_info))
        mlb = MultiLabelBinarizer()
        encoded_info = mlb.fit_transform(road_info)
        encoded_info_roads.append(encoded_info)
        # print((encoded_info_roads[0]))
    # print(mlb.classes_)

    # Combine the encoded informational variables with X_narratives
    feature_matrices = [X_narratives]
    for encoded_info in encoded_info_roads:
        encoded_info_sparse = sp.csr_matrix(encoded_info)
        feature_matrices.append(encoded_info_sparse)

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
# print(len(info))
# print(info.shape)
# print(RoadwaySystem.shape)
# print(info[0].shape)
# print(narrative_array.shape)
# print(classify.shape)
# print(classify)
# print(info.shape)
# print(info)
best_model, vectorizer, label_encoder = testing(narrative_array,classify,info)