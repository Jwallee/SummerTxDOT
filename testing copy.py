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

def encode_data(encoder, vectorizer, data, fields):
    print("Encoding data...")

# samples big dictionary
def take_random_sample(fields_array, classifications_array, narratives_array, size):
    import random
    indices = list(range(len(narratives_array)))

    # Shuffle the indices randomly
    random.shuffle(indices)

    # Take the first n indices from the shuffled list
    random_indices = indices[:size]

    fields_sample_array = [[] for i in range(len(fields_array))]
    classifications_sample_array = []
    narratives_sample_array = []

    for index in random_indices:
        # print(index)
        classifications_sample_array.append(classifications_array[index])
        narratives_sample_array.append(narratives_array[index])
        for i in range(len(fields_array)):
            fields_sample_array[i].append(fields_array[i][index])

    print("Random sample of size "+str(size)+" taken successfully.")
    return fields_sample_array, classifications_sample_array, narratives_sample_array, random_indices

# # # ACTUAL DATA PROCESSING STUFF
def testing(narratives, classifications, info_values, tests, classes, cv_value, model_folder, size):
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import MultiLabelBinarizer
    import scipy.sparse as sp
    import numpy as np
    import random
    import os
    import warnings
    import joblib
    import time


    start_time = time.time()
    model_loading = False
    # Filter out the specific warning
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl.styles.stylesheet")
    warnings.filterwarnings("ignore", category=UserWarning)

    if model_folder != '':
        file_names = os.listdir(model_folder)

        # Iterate over the file names
        file = 0
        for file_name in file_names:
            file_path = os.path.join(model_folder, file_name)

            # Check if the current item is a file
            if os.path.isfile(file_path):
                if file == 0:
                    pkl_encoder_path = file_path
                if file == 1:
                    txt_file_path = file_path
                elif file == 2:
                    pkl_model_path = file_path
                elif file == 3:
                    pkl_vectorizer_path = file_path
                file += 1

        # Print the file paths of the TXT and PKL files
        print("Narratives indices file path:", txt_file_path)
        print("PKL model file path:", pkl_model_path)
        print("PKL vectorizer file path:", pkl_vectorizer_path)
        print("PKL encoder file path:", pkl_encoder_path)

        with open(txt_file_path, "r") as file:
            content = file.read()

        # Remove the square brackets and split the remaining numbers
        numbers_str = content.strip()[1:-1]
        numbers_list = [int(number) for number in numbers_str.split(", ")]

        test_indices = numbers_list
        new_test_indices = []
        for i in range(len(narratives)):
            if i not in test_indices:
                new_test_indices.append(i)

        print(len(new_test_indices))
        test_narratives = []
        test_classifications = []
        test_info_values = [[] for _ in range(len(info_values))]
        random.shuffle(new_test_indices)
        for index in new_test_indices[:min(len(new_test_indices), tests)]:
            test_narratives.append(narratives[index])
            test_classifications.append(classifications[index])
            for i in range(len(info_values)):
                test_info_values[i].append(info_values[i][index])

        print("Random test sample of size " + str(tests) + " taken successfully.")

        model = joblib.load(pkl_model_path)
        vectorizer = joblib.load(pkl_vectorizer_path)
        encoder = joblib.load(pkl_encoder_path)
        model_loading = True

        X_narratives = vectorizer.transform(test_narratives)

        # Ensure all values in fields are strings
        fields = [[str(element) for element in row] for row in test_info_values]

        # Encode all values via MultiLabelBinarizer
        encoded_fields = [encoder.transform(array) for array in fields]

        # Combine the encoded informational variables with X_narratives
        feature_matrix = sp.hstack([X_narratives] + encoded_fields)

        # Get the model's feature size
        model_feature_size = model.n_features_in_

        print(feature_matrix.shape)

        # Check the number of features in feature_matrix
        if feature_matrix.shape[1] != model_feature_size:
            # Convert feature_matrix to csr_matrix for slicing
            feature_matrix = feature_matrix.tocsr()
            # Select common features between feature_matrix and model
            common_features = min(feature_matrix.shape[1], model_feature_size)
            feature_matrix = feature_matrix[:, :common_features]

        X_test = feature_matrix

        y_test = test_classifications
        y_pred = model.predict(X_test)


        import numpy as np
        y_pred_probs = model.predict_proba(X_test)
        y_pred_confidence = np.max(y_pred_probs, axis=1)

        confidence_threshold = 0.75
        # Tag test cases as confident or unconfident based on the threshold
        tagged_test_cases = []
        for i in range(len(test_indices)):
            if y_pred_confidence[i] >= confidence_threshold:
                tag = "Confident"
            else:
                tag = "Unconfident"
            tagged_test_cases.append((narratives[test_indices[i]], y_pred[i], classifications[test_indices[i]], tag))

        unconfident = []
        for a in range(0,len(test_indices)):
            if tagged_test_cases[a][-1] == "Unconfident":
                unconfident.append(a)

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

        output_file_path = 'output.txt'
        # Open the file in write mode
        with open(output_file_path, 'w') as file:
            # Redirect the print statements to both the terminal and the file
            print("Correct predictions:", correct_count)
            print("Correct predictions:", correct_count, file=file)
            print("Wrong predictions:", wrong_count)
            print("Wrong predictions:", wrong_count, file=file)
            accuracy = round(correct_count / len(y_test), 4)
            print("Accuracy:", accuracy, "[", correct_count, "out of", len(y_test), "]")
            print("Accuracy:", accuracy, "[", correct_count, "out of", len(y_test), "]", file=file)
            print("Confidence Correct:", confident_yes)
            print("Confidence Correct:", confident_yes, file=file)
            print("Confidence Wrong:", confident_no)
            print("Confidence Wrong:", confident_no, file=file)
            confidence_accuracy = confident_yes / (confident_yes + confident_no)
            print("Confidence Accuracy:", confidence_accuracy)
            print("Confidence Accuracy:", confidence_accuracy, file=file)
            confidence_rating = round((len(test_indices) - len(unconfident)) / len(test_indices), 2)
            print("Confidence Rating:", confidence_rating)
            print("Confidence Rating:", confidence_rating, file=file)

        # Print a message indicating the file has been saved
        print("Output saved to:", output_file_path)
        # print("Confidence Correct:", confident_yes)
        # print("Confidence Wrong:", confident_no)
        # print("Confidence Accuracy:",confident_yes/(confident_yes+confident_no))
        # print("Confidence Rating:", round((len(test_indices)-len(unconfident))/len(test_indices),2))

        # print(randoms1)
        # print(randoms2)
        
        # Return the best model
        return model, model_loading, test_indices, vectorizer, encoder



        

        

    else:
        import random
        indices = list(range(len(narratives)))

        # Shuffle the indices randomly
        random.shuffle(indices)

        # Take the first n indices from the shuffled list
        random_indices = indices[:size]

        fields_sample_array = [[] for i in range(len(info_values))]
        classifications_sample_array = []
        narratives_sample_array = []

        for index in random_indices:
            # print(index)
            classifications_sample_array.append(classifications[index])
            narratives_sample_array.append(narratives[index])
            for i in range(len(info_values)):
                fields_sample_array[i].append(info_values[i][index])

        print("Random sample of size "+str(size)+" taken successfully.")

        print("Recieved Data. Running testing funcion with "+str(len(narratives_sample_array))+" training datas and "+str(tests)+" tests.")


        # Vectorizes narratives via TfidVectorizer
        vectorizer = TfidfVectorizer(max_df=0.75, min_df=2, stop_words="english") #filters out words that appear in 95%, must occur in min 2 reports
        X_narratives = vectorizer.fit_transform(narratives_sample_array)

        # Ensures all values in fields are strings
        fields = [[str(element) for element in row] for row in fields_sample_array]

        # Encodes all values via MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        encoded_fields = [mlb.fit_transform(array) for array in fields]

        # Combine the encoded informational variables with X_narratives
        feature_matrix = [X_narratives]

        for encoded_info in encoded_fields:
            encoded_info_sparse = sp.csr_matrix(encoded_info)
            feature_matrix.append(encoded_info_sparse)

        X_combined = sp.hstack(feature_matrix)

        classifier = RandomForestClassifier()

        param_grid = {
        'n_estimators': [100],
        'max_depth': [100]
        }

        # Generating Random Test numbers to grab random narratives and classifications
        test_indices = random.sample(range(1, len(narratives_sample_array) - 1), tests)
        train_indices = [a for a in range(len(narratives_sample_array)) if a not in test_indices]
            
        # Creating Training data from the generated random numbers
        X_train = X_combined[train_indices]
        y_train = [classifications_sample_array[i] for i in train_indices]

        # Checking efficiencies of each parameter combinaion
        # cv = the number of tries it does for each parameter. more accuracy higher number, but takes a lot longer to process.
        grid_search = GridSearchCV(classifier, param_grid, cv=cv_value)
        grid_search.fit(X_train, y_train)

        # Get the best model from grid search
        best_model = grid_search.best_estimator_

        X_test = X_combined[test_indices]
        y_test = [classifications_sample_array[i] for i in test_indices]
        y_pred = best_model.predict(X_test)

        import numpy as np
        y_pred_probs = best_model.predict_proba(X_test)
        y_pred_confidence = np.max(y_pred_probs, axis=1)

        confidence_threshold = 0.75
        # Tag test cases as confident or unconfident based on the threshold
        tagged_test_cases = []
        for i in range(len(test_indices)):
            if y_pred_confidence[i] >= confidence_threshold:
                tag = "Confident"
            else:
                tag = "Unconfident"
            tagged_test_cases.append((narratives_sample_array[test_indices[i]], y_pred[i], classifications_sample_array[test_indices[i]], tag))

        unconfident = []
        for a in range(0,len(test_indices)):
            if tagged_test_cases[a][-1] == "Unconfident":
                unconfident.append(a)

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
        # Define the output file path
        

        output_file_path = 'output.txt'
        # Open the file in write mode
        end_time = time.time()
        execution_time = end_time - start_time  # Calculate the difference
    
        with open(output_file_path, 'w') as file:
            # Redirect the print statements to both the terminal and the file
            print("Training data size:", len(narratives_sample_array))
            print("Training data size:", len(narratives_sample_array), file=file)
            print("CV value:", cv_value)
            print("CV value:", cv_value, file=file)
            print("Correct predictions:", correct_count)
            print("Correct predictions:", correct_count, file=file)
            print("Wrong predictions:", wrong_count)
            print("Wrong predictions:", wrong_count, file=file)
            accuracy = round(correct_count / len(y_test), 4)
            print("Accuracy:", accuracy, "[", correct_count, "out of", len(y_test), "]")
            print("Accuracy:", accuracy, "[", correct_count, "out of", len(y_test), "]", file=file)
            print("Confidence Correct:", confident_yes)
            print("Confidence Correct:", confident_yes, file=file)
            print("Confidence Wrong:", confident_no)
            print("Confidence Wrong:", confident_no, file=file)
            confidence_accuracy = confident_yes / (confident_yes + confident_no)
            print("Confidence Accuracy:", confidence_accuracy, "[", confident_yes, "out of", confident_yes+confident_no, "]")
            print("Confidence Accuracy:", confidence_accuracy, "[", confident_yes, "out of", confident_yes+confident_no, "]", file=file)
            confidence_rating = round((len(test_indices) - len(unconfident)) / len(test_indices), 2)
            print("Confidence Rating:", confidence_rating)
            print("Confidence Rating:", confidence_rating, file=file)
            print(f"Execution time for training: {execution_time} seconds", file=file)

        # Print a message indicating the file has been saved
        print("Output saved to:", output_file_path)

        # print(randoms1)
        # print(randoms2)
        
        # Return the best model
        return best_model, model_loading, test_indices, vectorizer, mlb, output_file_path
# endregion