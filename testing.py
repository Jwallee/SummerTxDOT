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

# # # ACTUAL DATA PROCESSING STUFF
def testing(narratives,classifications,info_values,tests,classes):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import scipy.sparse as sp
    import numpy as np
    import random
    import warnings
    from openpyxl import Workbook
    # Filter out the specific warning
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl.styles.stylesheet")

    print("Recieved Data. Running testing funcion with "+str(len(narratives))+" training datas and "+str(tests)+" tests.")

    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()

    # Transforms the narratives to numerical data that is then analyzed by SciKit.
    X_narratives = vectorizer.fit_transform(narratives)
    # print(X_narratives.shape)
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

    info_types = [[str(value) for value in sublist] for sublist in info_types]
    encoded_info_roads = []
    mlb = MultiLabelBinarizer()
    for a in info_types:
        encoded_info = mlb.fit_transform(a)
        encoded_info_roads.append(encoded_info)

    feature_matrices = [X_narratives]
    for encoded_info in encoded_info_roads:
        feature_matrices.append(encoded_info)
            
    # Converts encoded data into a data matrix
    X_combined = sp.hstack(feature_matrices)

    # Instantiate and train the RandomForestClassifier
    classifier = RandomForestClassifier()

    # Create a testing grid.
    # n_estimators = Forest Size
    # max_depth = Number of decisions each forest has to make
    param_grid = {
    'n_estimators': [200, 300, 500, 600, 800],
    'max_depth': [25, 50, 75, 100, 125]
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
    X_train = sp.vstack([X_combined.getrow(i) for i in others])
    y_train = [classifications[i] for i in others]
    y_train = np.array(y_train)
    # y_train = classifications[others]

    # Checking efficienceies of each parameter combinaion
    # cv = the number of tries it does for each parameter. more accuracy higher number, but takes a lot longer to process.
    grid_search = GridSearchCV(classifier, param_grid, cv=7)
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    X_test = sp.vstack([X_combined.getrow(i) for i in randoms])
    y_test = [classifications[i] for i in randoms]
    y_test = np.array(y_test)
    # y_test = classifications[randoms]
    y_pred = best_model.predict(X_test)

    for a in range(0,len(randoms)):
        print("Narrative:", narratives[randoms[a]])
        print("Predicted Label:", classes[y_pred[a]])
        print("Actual Label:", classes[classifications[randoms[a]]])
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