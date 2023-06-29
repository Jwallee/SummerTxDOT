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
'''
Region Definitions:

Storing Data Example: Road Class/CityStreet

Field (string) - Name of where all of the information is stored (I organized it by what im finding i.e. Road Class, Severity)
specs (Array) - Contains all of the classification names that will be accessed via filepath (i named my folders CityStreet, CountyRoad etc.)
classifiers (Array) - Contains all of the classifiers that the machine learning module will use to link up with the narratives or whatever you link to.

RoadwaySystem (list) - Contains all info for Roadway System interpretive field
OutsideCL (list) - Contains all info for Outside City Limits interpretive field
TollRoad (list) - Contains all info for Toll Road interpretive field
info (array) - Contains all of the above lists and allows the machine learning function to apply these conditions to the linking process
columns (list) - Column names for where you're reading information into the above lists from an excel spreadsheet
dataset_limit (int) - An integer limiting the number of datapoints to 50. Can be changed if you have longer spreadsheets and want more data processed.
'''


Field = 'Road Class'
specs = np.array(["CityStreet","CountyRoad","FarmToMarket","Interstate","NonTrafficway","OtherRoads","Tollway","US&StateHighways"])
classifiers = np.array(["CITY STREET", "COUNTY ROAD", "FARM TO MARKET", "INTERSTATE", "NON-TRAFFICWAY", "OTHER ROADS", "TOLLWAY", "US & STATE HIGHWAYS"])

RoadwaySystem = []
OutsideCL = []
TollRoad = []
info = np.array([])
columns = ['B','C','E']
dataset_limit = 50

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

# Create info array and reshape it
info = np.array([RoadwaySystem, OutsideCL, TollRoad])
info = info.reshape(3, 341)
# endregion

# region READING TEXT INTO PROGRAM
'''
Region Definitions:

narrative_array (array) - Takes all the narratives and puts them into one long array. (row x columns) system.
classify (array) - Automatically makes a long array containing classifiers matching to the narrative array size.
'''
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


classify = np.array([])
for run in range(len(classifiers)):
    for assign in range(sizer[run]):
        classify = np.append(classify, classifiers[run])

# endregion


# region MACHINE LEARNING MODULE
'''
Region Definitions:

sklearn - SciKit library, which is responsible for the machine learning algorithms and classifiers

def testing(narratives,classifications,info_values):
    - Testing is a function that takes in narrative information, proper matching classifications and any information values (worked on above) that can help possibly improve the learning algorithm.
    Inputs:
        narratives (array) - Input array of all the narratives (MUST BE 1D)
        classifications (array) - Input array of all the classifications (MUST BE 1D and same size as narratives)
        info_values (array) - A dimensional array divided by informational values. For example, here we look at three informational values (RoadwaySystem, OutsideCL, TollRoad), so the initial size of info_values is 3.
            Within each of those three values are 341 (the number of narratives) pieces of information that match the narratives and classifiers.
'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import scipy.sparse as sp
# # # ACTUAL DATA PROCESSING STUFF

def testing(narratives,classifications,info_values):
    # Checking sizes (EXTRA CODE FOR CHECKING, NOT NEEDED)
    # print(len(info_values))
    # print(len(info_values[0]))
    # print(len(info_values[1]))
    # print(len(info_values[2]))
    # print(len(classifications))
    # print(len(narratives))

    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()

    # Transforms the narratives to numerical data that is then analyzed by SciKit.
    X_narratives = vectorizer.fit_transform(narratives)
    # print(vectorizer.vocabulary_)

    # Prepare the data for MultiLabelBinarizer. This takes your information and makes it a string with number of dimensions categorized by the informational values.
    info_types = []
    for road_info in info_values:
        road_types = []
        for type_data in road_info:
            road_types.append(type_data)
        info_types.append(road_types)


    # Encode the informational variables
    # This loop takes the information in info_types and encodes based on each grouped informational value.
    # This can then be processed by SciKit.
    encoded_info_roads = []
    for road_info in info_types:
        mlb = MultiLabelBinarizer()
        encoded_info = mlb.fit_transform(road_info)
        encoded_info_roads.append(encoded_info)

    # Combine the encoded informational variables with X_narratives
    feature_matrices = [X_narratives]
    for encoded_info in encoded_info_roads:
        encoded_info_sparse = sp.csr_matrix(encoded_info)
        feature_matrices.append(encoded_info_sparse)

    # Converts encoded data into a data matrix
    X_combined = sp.hstack(feature_matrices)

    # Split the data into training and testing sets.
    # This automatically takes the combined matrix and classifications and creates a testing set out of 20 percent of the data entered. random_state = 42 just tells the code to generate test cases randomly.
    X_train, X_test, y_train, y_test = train_test_split(X_combined, classifications, test_size=0.2, random_state=42)
    
    # Instantiate and train the RandomForestClassifier
    classifier = RandomForestClassifier()


    # Create a testing grid.
    # n_estimators = Forest Size
    # max_depth = Number of decisions each forest has to make
    param_grid = {
    'n_estimators': [200, 300, 500, 600, 800],
    'max_depth': [25, 50, 75, 100, 125]
    }

    # Checking efficienceies of each parameter combinaion
    # cv = the number of tries it does for each parameter. more accuracy higher number, but takes a lot longer to process.
    grid_search = GridSearchCV(classifier, param_grid, cv=6)
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Make predictions on the testing set
    y_pred = best_model.predict(X_test)

    # Print narratives, predicted labels, and actual labels
    for narrative, pred_label, actual_label in zip(narratives, y_pred, y_test):
        print("Narrative:", narrative)
        print("Predicted Label:", pred_label)
        print("Actual Label:", actual_label)
        print()
        
    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    
    # Return the best model
    return best_model, vectorizer
# endregion


# RUNNING THE CODE
best_model, vectorizer = testing(narrative_array,classify,info)