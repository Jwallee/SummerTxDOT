import pandas as pd
import json
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Load and preprocess data
    data = request.get_json()
    df = preprocess_data(data)

    # Convert DataFrame to a list of lists for TensorFlow Serving
    data_for_model = df.values.tolist()
    request_data = json.dumps({"instances": data_for_model})
    
    # Make POST request to TensorFlow Serving
    response = make_tf_serving_request(request_data)
    
    if response.ok:
        predictions = response.json()['predictions']
        return jsonify(predictions), 200
    else:
        return f"Error: {response.status_code} {response.text}", response.status_code

def preprocess_data(data):
    df = pd.DataFrame(data)
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

    # Normalize numerical columns in the dataset to be between 0 and 1
    data_features_normalized = data_features_no_missing.copy()
    numerical_columns = data_features_normalized.select_dtypes(include=['float64', 'int64']).columns
    data_features_normalized[numerical_columns] = (data_features_normalized[numerical_columns] - data_features_normalized[numerical_columns].min()) / (data_features_normalized[numerical_columns].max() - data_features_normalized[numerical_columns].min())

    # Encode categorical columns in the dataset using one-hot encoding
    data_features_encoded = pd.get_dummies(data_features_normalized)   # One-hot encoding

    # Normalize the test data
    test_features_normalized = data_features_encoded.copy()
    test_features_normalized[numerical_columns] = (test_features_normalized[numerical_columns] - test_features_normalized[numerical_columns].min()) / (test_features_normalized[numerical_columns].max() - test_features_normalized[numerical_columns].min())

    # One-hot encode the test data
    df = pd.get_dummies(test_features_normalized)

    return df

def make_tf_serving_request(request_data):
    tf_serving_url = 'http://tensorflow-serving:8501/v1/models/model:predict'  # Adjust the model name if necessary
    response = requests.post(tf_serving_url, data=request_data)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)