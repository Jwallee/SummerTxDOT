# **Machine Learning for Interpreted Fields**

*Property of TxDOT -  Unauthorized distribution, reproduction, or use of the code in any form without explicit permission from the author is strictly prohibited.*

Creators:

**Aiden Seibel and James Robinett**

*Updated: 10/22/23*

---
## Description

This code is meant to be used to process crash fields and narratives from the TxDOT CR-3 Texas peace officer's crash report form with the purpose of predicting interpreted fields that have to be manually filled out. We can train a machine learning algorithm on the data collected by TxDOT to create models which can predict interpreted fields. This will be done in a few steps:

1. Download and compile all of the neccessary data for a given field
2. Preprocess the data
3. Train a model on the data
4. Use the information to predict the fields that need to be filled out

**What is the CR-3 report?**

The CR-3 Texas peace officer's crash report form is filled out by officers at the scene of a crash that occurred on publicly maintained roads and exceeded $1,000 dollars in property damage. It contains information about the crash, the vehicles involved, and the people involved. The report is then sent to TxDOT where it is processed and the information is put into a database, which is then used to analyze the crashes and see what can be done to prevent them in the future.

**What is an investigator narrative?**

A narrative is a section of the CR-3 report that is filled out by the officer. It contains information about the crash that is not included in the other sections of the report. This information is very useful for analysis and can be used to predict the fields that need to be filled out.

**What is machine learning?**

Machine learning is a type of artificial intelligence that uses data to make predictions. It is used in many different fields and is very useful for making predictions based on data.

---

## **Files**

`PipelineHandler.py` is a Python module that simplifies the creation and management of machine learning pipelines for classification tasks.

*Key Features:*

* Flexible Data Handling: Easily import and preprocess training and testing data from CSV files.
* Customizable Pipelines: Build machine learning pipelines with configurable preprocessing steps and classifier options.
* Cross-Validation: Evaluate model performance using cross-validation to ensure robustness.
* Confidence Thresholding: Determine a confidence threshold for predictions and assess the model's confidence accuracy.
* Feature Importance: Calculate feature importances to understand the impact of input features on model accuracy.
* Confusion Matrix Visualization: Visualize confusion matrices for both overall predictions and confident predictions.

`MultiPipelineHandler.py` is a Python module designed to handle multiple PipelineHandler instances, each corresponding to a specific interpreted field. Its primary purpose is to provide a unified interface for managing a collection of classifiers, making it easier to work with machine learning pipelines for various data attributes in the context of crash prediction.

*Key Features:*

* Interpreted Field Handling: Organizes and manages individual PipelineHandler instances for different interpreted fields.
* Classifier Import: Allows importing pre-built classifiers from files to expedite prediction tasks.
* Data Management: Facilitates setting data frames for training and testing across all pipelines.
* Training and Testing: Provides methods for training and testing all pipelines collectively.
* Crash and Unit Prediction: Predicts interpreted fields for a given crash or unit and exports predictions to XML files.
* Export and Import: Enables exporting and importing models for efficient sharing and reuse.

`InterpretedField.py` is a Python module designed to represent and manage interpreted fields related to crash data. It provides a structured way to handle the attributes and features associated with different interpreted fields, making it easier to work with machine learning pipelines for crash prediction.

*Key Features:*

* Interpreted Field Representation: Allows you to create instances of InterpretedField objects, each corresponding to a specific interpreted field, with attributes like name, vectorized features, encoded features, classification feature, and unit-based indicator.

* Flexible Feature Assignment: Provides methods to set vectorized and encoded features either manually or randomly based on a given feature list. This flexibility aids in customizing the preprocessing steps for machine learning pipelines.

* Error Handling: Handles cases where the provided field name does not match any preset dictionaries, ensuring that the object can still be initialized with default values.

* Preset Dictionaries: Includes preset dictionaries that map field names to their respective vectorized features, encoded features, classification features, and unit-based indicators. These dictionaries help automate the process of setting features for specific fields.

*Other Parts:*

* Unit Columns: Lists unit-related columns that can be associated with crash data, making it easier to manage attributes related to units in crash prediction.

* Crash Columns: Lists crash-related columns that can be associated with crash data, providing a structured way to handle attributes related to crashes.

`feature_selection.py` is a Python script designed for the purpose of feature selection and importance calculation for different interpreted fields. It utilizes the PipelineHandler and InterpretedField classes to manage the process of selecting relevant features and calculating their importance in the context of crash prediction.

*Key Features:*

* Feature Selection: The script selects a set of relevant features for each interpreted field, considering whether the field is unit-based or crash-based. It dynamically chooses columns from the available data based on the field's nature.

* Model Training and Testing: The script uses the selected features to train and test machine learning models for crash prediction. It leverages the PipelineHandler class to handle these tasks efficiently.

* Feature Importance Calculation: After training, the script calculates feature importances for each selected feature. It measures the impact of each feature on the prediction task.

* Data Export: The script exports the calculated feature importances to separate CSV files for each interpreted field, making it easy to analyze and visualize the results.

* Automation: The script automates the process of feature selection, model training, and feature importance calculation for multiple interpreted fields, streamlining the analysis of different attributes in the context of crash prediction.

* Execution Time Tracking: The script tracks and displays the time taken for each field's feature selection and importance calculation, providing insights into the computational effort required.

`DataLoader.py` is a Python script designed to compile and prepare data from various sources into CSV format. Its primary purpose is to streamline the data preprocessing step, making it easier to work with combined data in the context of data analysis and modeling.

*Key Features:*

* Data Compilation: Combines data from narratives, crash fields, unit fields, and property damages into a unified dictionary structure.
* Column Mapping: Provides dictionaries (crash_column_number_dict and unit_column_number_dict) for mapping column numbers to column names, facilitating data export with correct headers.
* CSV Export: Generates CSV files with appropriate headers and writes data rows for both crash and unit data, ensuring data integrity and consistency.
* Handling Missing Data: Filters out crashes with missing values, ensuring that only complete data entries are included in the exported CSV files.
* Example Usage: Demonstrates how to use the DataLoader class to compile and export data efficiently from specified file paths.