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
def testing(narratives,classifications,info_values,tests,classes,cv_value):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import scipy.sparse as sp
    import numpy as np
    import random
    import warnings
    # Filter out the specific warning
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl.styles.stylesheet")
    warnings.filterwarnings("ignore", category=UserWarning)

    print("Recieved Data. Running testing funcion with "+str(len(narratives))+" training datas and "+str(tests)+" tests.")

    # Initialize the CountVectorizer
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english")

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
    'n_estimators': [600, 800],
    'max_depth': [100, 125]
    }

    randoms = random.sample(range(1, len(narratives) - 1), tests)

    others = [a for a in range(len(narratives)) if a not in randoms]
        
    # Creating Training data from the generated random numbers
    X_train = sp.vstack([X_combined.getrow(i) for i in others])
    y_train = [classifications[i] for i in others]
    y_train = np.array(y_train)
    # y_train = classifications[others]

    # Checking efficienceies of each parameter combinaion
    # cv = the number of tries it does for each parameter. more accuracy higher number, but takes a lot longer to process.
    grid_search = GridSearchCV(classifier, param_grid, cv=cv_value)
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    X_test = sp.vstack([X_combined.getrow(i) for i in randoms])
    y_test = [classifications[i] for i in randoms]
    y_test = np.array(y_test)
    y_pred = best_model.predict(X_test)
    y_pred_probs = best_model.predict_proba(X_test)
    y_pred_confidence = np.max(y_pred_probs, axis=1)

    confidence_threshold = 0.75
    # Tag test cases as confident or unconfident based on the threshold
    tagged_test_cases = []
    for i in range(len(randoms)):
        if y_pred_confidence[i] >= confidence_threshold:
            tag = "Confident"
        else:
            tag = "Unconfident"
        tagged_test_cases.append((narratives[randoms[i]], classes[y_pred[i]], classes[classifications[randoms[i]]], tag))

    unconfident = []
    for a in range(0,len(randoms)):
        print("Narrative:", narratives[randoms[a]])
        print("Predicted Label:", classes[y_pred[a]])
        print("Actual Label:", classes[classifications[randoms[a]]])
        if tagged_test_cases[a][-1] == "Unconfident":
            unconfident.append(a)
        print("Tag:", tagged_test_cases[a][-1])
        print()

    # Evaluate the model's performance
    correct_count = 0
    wrong_count = 0

    confident_yes = 0
    confident_no = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            if i in unconfident:
                pass
            else:
                confident_yes += 1
            correct_count += 1
        else:
            if i in unconfident:
                pass
            else:
                confident_no += 1
            wrong_count += 1

    print("Correct predictions:", correct_count)
    print("Wrong predictions:", wrong_count)
    accuracy = correct_count/len(y_test)
    accuracy = round(accuracy, 2)
    print("Accuracy:", accuracy,"[",correct_count,"out of",len(y_test),"]")
    print("Confidence Correct:", confident_yes)
    print("Confidence Wrong:", confident_no)
    ave = confident_yes+confident_no
    print("Confidence Accuracy:",confident_yes/ave)

    precision = precision_score(y_test, y_pred, average='weighted')
    print("Precision:", round(precision,2))

    print("Unconfident in these test cases: ",unconfident)
    print("Confidence Rating:", round((len(randoms)-len(unconfident))/len(randoms),2))
    
    # Return the best model
    return best_model, vectorizer
# endregion