#region Request
# For data preprocessing
# We are going to assume that the XML file is already loaded into a json request, and here is where we load it into a pandas dataframe
"""
The json request will look like this:
"""
import pandas as pd
import json

def load_xml(json_request):
    # Load the json request into a pandas dataframe
    df = pd.DataFrame(json_request)
    return df

# Load json file /Users/grantrobinett/2024/coding/txdot_interpreted_fields/test_improvements/dataLocal/crash_data_1000.json
with open('/Users/grantrobinett/2024/coding/txdot_interpreted_fields/test_improvements/dataLocal/crash_data_1000.json') as f:
    json_request = json.load(f)

df = load_xml(json_request)
print(df.head())
#endregion

#region Processing
"""
Dataframe preprocessing:
- Ensure that the column names are in the correct format
- Ensure that the data types are correct
- Ensure that the data is formatted so the model can understand it
- Encode the data so the model can understand it
"""

# If index column exists, drop it
if 'index' in df.columns:
    df = df.drop(columns='index')

# List of columns to be predicted and therefore removed from the dataset
columns_to_predict = [
    "Road_Relat_ID", "Intrsct_Relat_ID", "Road_Cls_ID", "Harm_Evnt_ID",
    "FHE_Collsn_ID", "Obj_Struck_ID", "Phys_Featr_1_ID", "Phys_Featr_2_ID",
    "Bridge_Detail_ID", "Othr_Factr_ID", "Road_Part_Adj_ID", "Investigator_Narrative"
]

# Drop the columns to be predicted from the dataset
data_features = df.drop(columns=columns_to_predict)

# Drop columns with any missing values
data_features_no_missing = data_features.dropna(axis=1)

# Print size of the dataset after dropping columns with missing values
print("\nSize of the dataset after dropping columns with missing values:")
print(data_features_no_missing.shape)

# Normalize numerical columns in the dataset to be between 0 and 1
data_features_normalized = data_features_no_missing.copy()
numerical_columns = data_features_normalized.select_dtypes(include=['float64', 'int64']).columns
data_features_normalized[numerical_columns] = (data_features_normalized[numerical_columns] - data_features_normalized[numerical_columns].min()) / (data_features_normalized[numerical_columns].max() - data_features_normalized[numerical_columns].min())

# Print summary statistics of the normalized dataset
print("\nSummary statistics of the normalized dataset:")
print(data_features_normalized.describe())

# Encode categorical columns in the dataset using one-hot encoding
# Print size of the dataset before one-hot encoding
print("\nSize of the dataset before one-hot encoding:")
print(data_features_normalized.shape)
data_features_encoded = pd.get_dummies(data_features_normalized)   # One-hot encoding
# Print size of the dataset after one-hot encoding
print("\nSize of the dataset after one-hot encoding:")
print(data_features_encoded.shape)

# Normalize the test data
test_features_normalized = data_features_encoded.copy()
test_features_normalized[numerical_columns] = (test_features_normalized[numerical_columns] - test_features_normalized[numerical_columns].min()) / (test_features_normalized[numerical_columns].max() - test_features_normalized[numerical_columns].min())

# One-hot encode the test data
test_features_encoded = pd.get_dummies(test_features_normalized)

# Print size of the test data
print("\nSize of the test data:")
print(test_features_encoded.shape)

#endregion

#region Inference Server Request
"""
Data Preprocessed. Able to send to inference server.
We will be using the test_features_encoded dataframe to send to the inference server.
- Input: test_features_encoded converted to json request
- Output: json response of the model's predictions
"""
import requests

# Convert DataFrame to a list of lists for TensorFlow Serving
data = test_features_encoded.values.tolist()

# Format the POST request data
request_data = json.dumps({"instances": data})

# Specify the TensorFlow Serving REST API endpoint
tf_serving_url = 'http://localhost:8501/v1/models/model:predict'  # Adjust the model name if necessary

# Send the POST request
response = requests.post(tf_serving_url, data=request_data)

# Check the response
if response.status_code == 200:
    # Load the JSON response
    predictions = response.json()['predictions']
    # Do something with the predictions
    print(predictions)
else:
    print("Error:", response.status_code, response.text)

#endregion


