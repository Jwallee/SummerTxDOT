from flask import Flask, request, jsonify
import pandas as pd
import json
import requests

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    json_request = request.get_json()
    df = pd.DataFrame(json_request)
    
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

    data = df.values.tolist()
    request_data = json.dumps({"instances": data})
    tf_serving_url = 'http://tensorflow-serving:8501/v1/models/model:predict'
    response = requests.post(tf_serving_url, data=request_data)
    if response.status_code == 200:
        predictions = response.json()['predictions']
        return jsonify({"predictions": predictions}), 200
    else:
        return jsonify({"error": response.text}), response.status_code

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
