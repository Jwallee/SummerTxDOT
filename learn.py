# region MACHINE LEARNING MODULE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
# from load_data import load_data, take_random_sample
import scipy.sparse as sp
import random, time
import presets
from sklearn.pipeline import Pipeline
import joblib

def learn_crash(texts,classifications,extra_data,model_path,n_estimators_values, max_depth_values, cv_values):
    start = time.time()

    # Convert any numerical values in extra_data to strings
    extra_data = [[str(item) for item in extras] for extras in extra_data]

    # Create a list to store the combined data
    combined_data = []

    # Combine each text with its corresponding extra data
    for i in range(len(texts)):
        text = texts[i]
        extras_for_text = [str(item) for item in [extra[i] for extra in extra_data]]
        combined_data.append(text + " " + " ".join(extras_for_text))

    # Vectorize the narratives using TfidfVectorizer and add to the list of feature transformers
    vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, stop_words="english")

    # Create a logistic regression classifier
    classifier = RandomForestClassifier()

    # Build the pipeline by combining feature transformers and classifier
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('clf', classifier)
    ])

     # Set up the parameter grid for GridSearchCV
    param_grid = {
        'clf__n_estimators': n_estimators_values,
        'clf__max_depth': max_depth_values,
    }

    # Perform grid search using cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv_values, n_jobs=-1)
    grid_search.fit(combined_data, classifications)

    # Get the best hyperparameters from the grid search
    best_n_estimators = grid_search.best_params_['clf__n_estimators']
    best_max_depth = grid_search.best_params_['clf__max_depth']

    # Update the pipeline with the best hyperparameters
    pipeline.set_params(clf__n_estimators=best_n_estimators, clf__max_depth=best_max_depth)

    # Fit the pipeline with the combined data and corresponding classifications
    pipeline.fit(combined_data, classifications)
    # Save the pipeline for later use
    joblib.dump(pipeline, model_path)

    # Evaluate the model's performance
    total_time_taken = time.time() - start

    print("Best n_estimators:", best_n_estimators)
    print("Best max_depth:", best_max_depth)

    print("Completed in", total_time_taken, "seconds.")

# endregion

def run_model(sample_size: int, test_percent: float, cv: int, fields_array_: list[any], classifications_array_: list[any], narratives_array_: list[str], model_path, model_load):
    if model_load == False:
        percent = float(test_percent/100)

        train_size = int(len(narratives_array_)*(1-percent))

        train_text = [narratives_array_[i] for i in range(train_size)]
        train_class = [classifications_array_[i] for i in range(train_size)]
        train_extra = [row[:train_size] for row in fields_array_]

        test_text = [narratives_array_[i] for i in range(train_size, len(narratives_array_))]
        test_class = [classifications_array_[i] for i in range(train_size, len(classifications_array_))]
        test_extra = [row[train_size:] for row in fields_array_]
        

        learn_crash(train_text, train_class, train_extra, model_path, n_estimators_values=[100, 200, 300], max_depth_values=[100, 200, 300], cv_values=cv)
        print("Model saved to", model_path)

        "Test the exported pipeline"
        pipeline = joblib.load(model_path)
        print("Model loaded from", model_path)

        # Convert any numerical values in extra_data to strings
        extra_test = [[str(item) for item in extras] for extras in test_extra]

        # Create a list to store the combined data
        combined_data = []

        # Combine each text with its corresponding extra data
        for i in range(len(test_text)):
            text = test_text[i]
            extras_for_text = [str(item) for item in [extra[i] for extra in extra_test]]
            combined_data.append(text + " " + " ".join(extras_for_text))

        # # Test the pipeline
        predictions = pipeline.predict(combined_data)
        probabilities = pipeline.predict_proba(combined_data)

        

        "Count the number of correct predictions"
        print("Correct predictions:", sum([1 if predictions[i] == test_class[i] else 0 for i in range(len(predictions))]))
        "Count the number of incorrect predictions"
        print("Incorrect predictions:", sum([1 if predictions[i] != test_class[i] else 0 for i in range(len(predictions))]))
        "Print accuracy"
        print("Accuracy:", sum([1 if predictions[i] == test_class[i] else 0 for i in range(len(predictions))]) / len(predictions))

        # Print the number of tests with confidence over 0.75
        count_confidence_over_075 = 0
        for i, prediction in enumerate(predictions):
            confidence = probabilities[i][pipeline.classes_.tolist().index(prediction)]
            if confidence > 0.75:
                count_confidence_over_075 += 1

        print("Number of tests with confidence over 0.75:", count_confidence_over_075)

        # Print the number of correct predictions with confidence over 0.75
        count_correct_confidence_over_075 = 0
        for i, prediction in enumerate(predictions):
            true_label = test_class[i]
            confidence = probabilities[i][pipeline.classes_.tolist().index(prediction)]

            if confidence > 0.75 and prediction == true_label:
                count_correct_confidence_over_075 += 1

        print("Number of correct predictions with confidence over 0.75:", count_correct_confidence_over_075)

        print("Confidence Accuracy:", count_correct_confidence_over_075 / count_confidence_over_075)

    else:
        pipeline = joblib.load(model_path)
        print("Model loaded from", model_path)

        # Convert any numerical values in extra_data to strings
        extra_test = [[str(item) for item in extras] for extras in fields_array_]

        # Create a list to store the combined data
        combined_data = []

        # Combine each text with its corresponding extra data
        for i in range(len(narratives_array_)):
            text = narratives_array_[i]
            extras_for_text = [str(item) for item in [extra[i] for extra in extra_test]]
            combined_data.append(text + " " + " ".join(extras_for_text))

        # # Test the pipeline
        predictions = pipeline.predict(combined_data)
        probabilities = pipeline.predict_proba(combined_data)

        

        "Count the number of correct predictions"
        print("Correct predictions:", sum([1 if predictions[i] == classifications_array_[i] else 0 for i in range(len(predictions))]))
        "Count the number of incorrect predictions"
        print("Incorrect predictions:", sum([1 if predictions[i] != classifications_array_[i] else 0 for i in range(len(predictions))]))
        "Print accuracy"
        print("Accuracy:", sum([1 if predictions[i] == classifications_array_[i] else 0 for i in range(len(predictions))]) / len(predictions))

        # Print the number of tests with confidence over 0.75
        count_confidence_over_075 = 0
        for i, prediction in enumerate(predictions):
            confidence = probabilities[i][pipeline.classes_.tolist().index(prediction)]
            if confidence > 0.75:
                count_confidence_over_075 += 1

        print("Number of tests with confidence over 0.75:", count_confidence_over_075)

        # Print the number of correct predictions with confidence over 0.75
        count_correct_confidence_over_075 = 0
        for i, prediction in enumerate(predictions):
            true_label = classifications_array_[i]
            confidence = probabilities[i][pipeline.classes_.tolist().index(prediction)]

            if confidence > 0.75 and prediction == true_label:
                count_correct_confidence_over_075 += 1

        print("Number of correct predictions with confidence over 0.75:", count_correct_confidence_over_075)

        print("Confidence Accuracy:", count_correct_confidence_over_075 / count_confidence_over_075)


