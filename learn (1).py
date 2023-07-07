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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_processing.load_data import load_data, take_random_sample
import scipy.sparse as sp
import random
import time 

# # # ACTUAL DATA PROCESSING STUFF
def testing(narratives,classifications,info_values, tests, cv, log_output=True):
    # Checking sizes (EXTRA CODE FOR CHECKING, NOT NEEDED)
    # print(len(info_values))
    # print(len(info_values[0]))
    # print(len(info_values[1]))
    # print(len(info_values[2]))
    # print(len(classifications))
    # print(len(narratives))

    start = time.time()

    # Initialize the CountVectorizer
    vectorizer = TfidfVectorizer()
    if log_output == True:
        print("Initialized Count Vectorizer.")

    # Transforms the narratives to numerical data that is then analyzed by SciKit.
    X_narratives = vectorizer.fit_transform(narratives)
    # print(X_narratives.shape)
    # print(vectorizer.vocabulary_)
    if log_output == True:
        print("Vectorized all narratives.")

    # Prepare the data for MultiLabelBinarizer. This takes your information and makes it a string with number of dimensions categorized by the informational values.
    info_types = []
    for road_info in info_values:
        road_types = []
        for type_data in road_info:
            road_types.append(str(type_data))
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

    if log_output == True:
        print("Encoded all data into data matrix.")

    # Instantiate and train the RandomForestClassifier
    classifier = RandomForestClassifier()
    if log_output == True:
        print("Initialized Random Forest Classifier.")

    # Create a testing grid.
    # n_estimators = Forest Size
    # max_depth = Number of decisions each forest has to make
    param_grid = {
    'n_estimators': [100],
    'max_depth': [100]
    }

    # Generating Random Test numbers to grab random narratives and classifications
    randoms = []
    for i in range(1,tests):
        randoms.append(random.randint(0, len(narratives)-1))
    
    # Everything not involved in testing is put into others
    others = []
    for a in range(len(narratives)):
        if a in randoms:
            a
        else:
            others.append(a)
        
    # Creating Training data from the generated random numbers
    X_train = X_combined[others]
    y_train = [classifications[i] for i in others]

    if log_output == True:
        print("Training data created. Checking parameter efficiencies...")

    # Checking efficiencies of each parameter combinaion
    # cv = the number of tries it does for each parameter. more accuracy higher number, but takes a lot longer to process.
    grid_search = GridSearchCV(classifier, param_grid, cv=cv)
    if log_output == True:
        print("Best parameters found. Training data...")
    grid_search.fit(X_train, y_train)

    if log_output == True:
        print("Training completed in " + str(time.time() - start) + " seconds.")

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    X_test = X_combined[randoms]
    y_test = [classifications[i] for i in randoms]
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    bad_count = 0

    for a in range(0,len(randoms)):
        if (max(y_proba[a]) > 0.50) and (y_pred[a] != classifications[randoms[a]]):
            print("Narrative:", narratives[randoms[a]])
            print("Fields:", end="\t")
            for i in info_values:
                print(i[randoms[a]], end=" ")
            print("\nPredicted Label:", y_pred[a])
            print("Actual Label:", classifications[randoms[a]])
            print("Class probabilities:", y_proba[a], end="\n\n")
            bad_count += 1

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy)

    if log_output == True:
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)
    
    time_taken = time.time() - start
    if log_output == True:
        print("Finished processes in " + str(time_taken) + " seconds.")
        print("Mislabeled crashes w/ confidence over 50%:", str(bad_count) + "/" + str(len(narratives)))
    print("Adjusted accuracy: " + str(accuracy + bad_count / len(narratives)))

    return best_model, vectorizer, time_taken
# endregion


def run_model(sample_size, test_percent, cv, fields_array_, classifications_array_, narratives_array_):

    fields_array_, classifications_array_, narratives_array_ = take_random_sample(fields_array_, classifications_array_, narratives_array_, sample_size)
    best_model, vectorizer, time_taken = testing(narratives_array_,classifications_array_,fields_array_, int(sample_size * test_percent), cv, log_output=True)

    print(time_taken)





# testing
file_path_to_narratives_file = "C:\\Users\\aseibel\\Documents\\investigator_narrative.csv"
file_path_to_fields_folder = "C:\\Users\\aseibel\\Documents\\extract_public_2023_20230629162130066_92029_20220101-20220630Texas\\crash"

# road class
# field_columns = [18,19,24,25,26,31]
# classification_column = 71

# # intersection related
field_columns = [32,39,41,49] 
classification_column = 66

fields_array_copy, classifications_array_copy, narratives_array_copy = load_data(file_path_to_narratives_file, file_path_to_fields_folder, field_columns, classification_column)

run_model(5000, 0.20, 3, fields_array_copy, classifications_array_copy, narratives_array_copy)
