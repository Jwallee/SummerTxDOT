from pandas.errors import DtypeWarning
import warnings, joblib, time
import pandas as pd
from InterpretedField import InterpretedField
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.inspection import permutation_importance

class PipelineHandler:
    """Handles a scikit-learn pipeline.
    
    Initialization
    ----------
    Training data filepath can be relative or absolute, but must correspond to a csv table with column headers.
    Testing data filepath, n_estimators and max_depth are optional; if no string is given for testing data, 
    the handler will use the training data. This comes with its consequences.

    ``handler = PipelineHandler(InterpretedField("intersection_related"), r"C:\\file\path\\to\\training\data", r"C:\\file\path\\to\\testing\data", 100, 100)``
    
    Attributes
    ----------
    ``interpreted_field: InterpretedField``
    ``training_df: DataFrame``
    ``testing_df: DataFrame``
    ``pipe: Pipeline``
    ``X: DataFrame``
    ``y: Series``
    ``predictions: ndarray``
    ``probabilities: ndarray``
    ``base_accuracy: float``
    ``conf_threshold: float``
    ``conf_percent: float``
    ``conf_accuracy: float``

    Methods
    ----------
    ``calculate_cv_score(cv: int)``
    ``sample_df()``
    ``train()``
    ``test()``
    ``calculate_base_accuracy()``
    ``calculate_confidence_threshold()``
    ``calculate_confidence_accuracy()``
    ``calculate_percent_confident()``
    ``calculate_class_accuracies()``
    ``predict_single_row()``
    ``export_to_filepath()``
    ``import_from_filepath()``
    ``calculate_feature_importances()``
    ``plot_feature_importances()``
    ``plot_confusion_matrix()``
    """

    def __init__(self, interpreted_field: InterpretedField, training_data_filepath: str = "", testing_data_filepath: str = "", use_bert: bool = False) -> None:
        """Creates a pipeline with TfidfVectorizer and OneHotEncoder.
        """
        self.interpreted_field = interpreted_field
        training_data_filepath = training_data_filepath
        testing_data_filepath = testing_data_filepath
        self.name = interpreted_field.name
    

    def set_data_frames(self, training_data_filepath: str = "", testing_data_filepath: str = ""):
        """Imports training and testing dataframes, ignores warning output log"""

        # select columns of interest from base data frame
        features_combined = list(set(self.interpreted_field.vectorized_features) | set(self.interpreted_field.encoded_features))
        features_combined.append(self.interpreted_field.classification_feature)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DtypeWarning)
            if training_data_filepath != "":
                self.training_df = pd.read_csv(training_data_filepath, encoding='windows-1252', dtype=str)
                self.training_df = self.training_df.loc[:, features_combined]
            if testing_data_filepath  != "":
                self.testing_df = pd.read_csv(testing_data_filepath, encoding='windows-1252', dtype=str)
                self.testing_df = self.testing_df.loc[:, features_combined]

        print("Successfully set data frames for "+self.name+":")
        print("Training data frame: "+training_data_filepath)
        print("Testing data frame: "+testing_data_filepath)

    def make_new_pipeline(self, n_estimators: int = 100, max_depth: int = 100):
        # ensure distinct vectorizers for each vectorized feature        
        tfidf_transformers = []
        for feature in self.interpreted_field.vectorized_features:
            tfidf_transformers.append((TfidfVectorizer(max_df=0.50, min_df=2, stop_words="english"), feature))

        # preprocessing steps for encoded features
        column_trans = make_column_transformer(
            (OneHotEncoder(handle_unknown='ignore'), self.interpreted_field.encoded_features), 
            *tfidf_transformers, 
            remainder='passthrough')

        # initialize classifier
        classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        # make the pipeline
        self.pipe = make_pipeline(column_trans, classifier)
        print("Pipeline created for "+str(self.name)+".")


    def calculate_cv_score(self, cv: int = 5):
        """Calculates how well the model generalizes to unseen data and how consistent its performance is across different data subsets. 
        Linear time complexity based on folds (cv).
        """        
        try:
            score = cross_val_score(self.pipe, self.X, self.y, cv=cv, scoring='accuracy').mean()
            print("CV score: "+str(round(score, 3)))
        except AttributeError as e:
            print(str(AttributeError))
            print("Cannot calculate CV score. Model is likely not trained.")
        return score


    def sample_df(self, sample_size: int = 10000, is_testing: bool = False, is_equalized: bool = False):
        """Samples the base data frame. Does not set self.X, self.y to samples automatically.
        """
        sampled_df = pd.DataFrame()

        # "equalized" sampling
        if is_equalized:
            # group the dataframe by the number of classifications
            if is_testing:
                df = self.testing_df.groupby(self.interpreted_field.classification_feature)
            else:
                df = self.training_df.groupby(self.interpreted_field.classification_feature)
            
            group_sample_size = sample_size / df.ngroups
            sizes = df.size()
            sampled_df = pd.DataFrame()

            for group_name, group_data in df:
                group_size = sizes[group_name]

                #sample from each group equally, if possible.
                try:
                    sample_size_per_group = int(group_sample_size)
                    if sample_size_per_group > group_size:
                        raise ValueError("Sample size per group is larger than the group size.")
                    sampled_group = group_data.sample(n=sample_size_per_group, replace=False)

                # not enough data, sample as much as possible
                except ValueError:
                    sampled_group = group_data.sample(n=group_size, replace=False)
                
                print("Sampled "+str(sampled_group.shape[0])+" from group "+ str(group_name))
                sampled_df = pd.concat([sampled_df, sampled_group])
            print("Sampled df has length of "+str(len(sampled_df)))

        # random sampling
        else:
            if is_testing:
                sampled_df = self.testing_df.sample(n=sample_size)
            else:
                sampled_df = self.training_df.sample(n=sample_size)
        
        #replace NAs with empty strings
        sampled_df.fillna('', inplace=True)

        # does not include the classification column
        features_combined = list(set(self.interpreted_field.vectorized_features) | set(self.interpreted_field.encoded_features))
        
        # separate features and classifications
        X = sampled_df.loc[:, features_combined]
        y = sampled_df[self.interpreted_field.classification_feature]
        
        print("Random sample of size", sampled_df.shape[0], "taken.")
        return X, y
    

    def train(self, sample_size: int = 10000):
        """Trains a model on a fresh sample from the base data frame.
        """
        start = time.time()
        # sample base data frame
        self.X, self.y = self.sample_df(sample_size, is_testing=False, is_equalized=True)

        print("Training model...")

        # training
        self.pipe.fit(self.X, self.y)
        print("Model successfully trained in "+str(round(time.time()-start, 2))+" seconds.")


    def test(self, sample_size: int = 10000, desired_conf_accuracy:float=0.96) -> dict[str, any]:
        """Tests a model on a sample from testing data and returns dictionary with base accuracy, confidence threshold, percent confident,
        and confidence accuracy.
        """
        
        # sample from base dataframe (equalized)
        self.X, self.y = self.sample_df(sample_size, is_testing=True, is_equalized=True)

        # predict on test data 
        self.predictions = self.pipe.predict(self.X) 
        self.probabilities = self.pipe.predict_proba(self.X)

        self.base_accuracy = self.calculate_base_accuracy()
        self.conf_threshold = self.calculate_confidence_threshold(desired_conf_accuracy)
        self.conf_percent = self.calculate_percent_confident(self.conf_threshold)
        self.conf_accuracy = self.calculate_confidence_accuracy(self.conf_threshold)

        return {"Base Accuracy": self.base_accuracy, 
                "Confidence Threshold": self.conf_threshold,
                "Percent Confident": self.conf_percent,
                "Confidence Accuracy": self.conf_accuracy}
    
    
    def calculate_base_accuracy(self) -> float:
        """Returns base accuracy only from the last test of a model.
        """
        correct = sum(a == p for a, p in zip(self.y, self.predictions))
        base_accuracy = correct / len(self.y)
        print("Base Accuracy: "+str(round(base_accuracy, 3))+" ("+str(correct)+"/"+str(len(self.y))+")")
        return base_accuracy


    def calculate_confidence_threshold(self, desired_conf_accuracy: float = 0.96) -> float:
        """Calculates confidence threshold via a binary search to achieved desired confidence accuracy
        """
        lower_bound = 0
        upper_bound = 1
        iterations = 0
        conf_threshold = 0.50
        while iterations < 7:
            conf_threshold = (lower_bound + upper_bound) / 2
            conf_accuracy = sum(self.probabilities[i][self.pipe.classes_.tolist().index(p)] > conf_threshold and p == a for i, (a, p) in enumerate(zip(self.y, self.predictions)))/sum([self.probabilities[i][self.pipe.classes_.tolist().index(p)] > conf_threshold for i, p in enumerate(self.predictions)])
            if conf_accuracy < desired_conf_accuracy:
                lower_bound = conf_threshold
            else:
                upper_bound = conf_threshold
            iterations += 1

        return conf_threshold


    def calculate_confidence_accuracy(self, conf_threshold: float) -> float:
        """Calculates confidence accuracy based on custom confidence threshold.
        """
        count_confident = sum([self.probabilities[i][self.pipe.classes_.tolist().index(p)] > conf_threshold for i, p in enumerate(self.predictions)])
        count_correct_confident = sum(self.probabilities[i][self.pipe.classes_.tolist().index(p)] > conf_threshold and p == a for i, (a, p) in enumerate(zip(self.y, self.predictions)))

        try:
            conf_accuracy = count_correct_confident/count_confident
        except ZeroDivisionError:
            conf_accuracy = 0

        print("Confidence Accuracy:", round(conf_accuracy, 3))
        return conf_accuracy

    
    def calculate_percent_confident(self, conf_threshold: float) -> float:
        """Calculates the proportion of confident predictions from custom threshold.
        """
        count_confident = sum([self.probabilities[i][self.pipe.classes_.tolist().index(p)] > conf_threshold for i, p in enumerate(self.predictions)])

        percent_confident = count_confident/len(self.y)
        print("% Confident: "+str(round(percent_confident, 3)))
        return percent_confident


    def calculate_class_accuracies(self) -> dict[str, float]:
        """Calculates the accuracy of the model on each classification of the data, similar to a confusion matrix.
        Returns results as a dictionary.
        """
        class_labels = set(self.y)
        class_accuracies = {}
        for label in class_labels:
            true_indices = self.y == label  # Indices where the true label matches the current class
            predicted_class_labels = self.predictions[true_indices]
            true_class_labels = self.y[true_indices]
            
            class_accuracy = accuracy_score(true_class_labels, predicted_class_labels)
            class_accuracies[label] = class_accuracy

        print("Accuracy for each class:")
        for label, accuracy in class_accuracies.items():
            print(f"Class {label}: {accuracy:.2f}")
        return class_accuracies


    def predict(self, data: pd.DataFrame, conf_threshold: float):
        """Predicts classification of a single row of data in a csv file.
        """
        features_combined = list(set(self.interpreted_field.vectorized_features) | set(self.interpreted_field.encoded_features))
        features = data.loc[:, features_combined]
        prediction = self.pipe.predict(features)[0]
        probabilities = self.pipe.predict_proba(features)[0]
        if max(probabilities) > conf_threshold:
            return {"Prediction": prediction, "Is_Confident": True}
        else:
            return {"Prediction": prediction, "Is_Confident": False}


    def export_to_filepath(self, filepath: str) -> bool:
        """Exports a model to the specified filepath. File size grows almost linear to the amount of training (~5mb per thousand training size)
        """
        try:
            joblib.dump(self.pipe, filepath)
            print("Successfully exported model to", filepath)
            return True
        
        #in case filepath doesn't exist, filepath is incorrect, etc
        except Exception as e:
            print("Failed to export model to specified filepath:", filepath)
            print(str(e))
            return False


    def import_from_filepath(self, filepath: str) -> bool:
        """Imports a model from the specified filepath.
        """
        try:
            self.pipe = joblib.load(filepath)
            print("Successfully imported model from", filepath)
            return True
        
        #in case filepath doesn't exist, filepath is incorrect, file can't be loaded
        except Exception as e:
            print("Failed to load model from", filepath)
            print(str(e))
            return False
        

    def calculate_feature_importances(self):
        """Calculates the importance of features in model accuracy. Does not require testing prior to being called.
        """

        warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread will likely fail.")

        # fetch feature names
        feature_names = list(set(self.interpreted_field.vectorized_features) | set(self.interpreted_field.encoded_features))
        
        # calculate feature importances
        result = permutation_importance(self.pipe, self.X, self.y, n_repeats=10, random_state=42, n_jobs=2)
        forest_importances = pd.Series(result.importances_mean, index=feature_names)
        return result, forest_importances


    def plot_feature_importances(self):
        """Plots feature importances as a histogram.
        """
        result, forest_importances = self.calculate_feature_importances()
        # create plot
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
        ax.set_title("Feature Importances for {} (via permutation on full model)".format(self.interpreted_field.name))
        ax.set_ylabel("Mean Accuracy Decrease")
        ax.set_xlabel("Feature")
        fig.tight_layout()
        
        plt.show()


    def plot_confusion_matrix(self):
        """Displays a confusion matrix based on the last test.
        """
        try:
            # graphics modules
            warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread will likely fail.")

            # create the confusion matrix for all predictions
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title("Confusion matrix for {}".format(self.interpreted_field.name))
            ConfusionMatrixDisplay.from_predictions(self.y, self.predictions, ax=ax)

            # determine confident predictions and their actual classifications using boolean mask trick 
            mask = self.probabilities.max(axis=1) >= self.conf_threshold
            confident = self.y[mask].tolist()
            confident_predictions = self.predictions[mask].tolist()

            # create the confusion matrix for confident predictions only
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.set_title("Confident confusion matrix for {}".format(self.interpreted_field.name))
            ConfusionMatrixDisplay.from_predictions(confident, confident_predictions, ax=ax2)

            plt.show()
            
        except AttributeError:
            print("Error: Model must be tested before confusion matrices can be built. Call test() before calculate_confusion_matrix() to resolve this issue.")
