from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained logistic regression model and the scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON input data from the request
    data = request.get_json(force=True)

    # Convert the input data into a pandas DataFrame
    # Ensure the order of columns matches the training data
    try:
        # Assuming input data is a dictionary or list of dictionaries
        input_df = pd.DataFrame(data, index=[0])
    except ValueError:
        # Handle case where input is a list of features directly
        input_df = pd.DataFrame([data])
    
    # Define the expected feature columns in the correct order
    # This should match the columns used to train the scaler and model
    expected_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df = input_df[expected_columns]

    # Preprocess the input data using the loaded StandardScaler
    input_scaled = scaler.transform(input_df)

    # Make predictions using the loaded logistic regression model
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # Return the prediction as a JSON response
    return jsonify({
        'prediction': int(prediction[0]),
        'prediction_probability_class_0': float(prediction_proba[0][0]),
        'prediction_probability_class_1': float(prediction_proba[0][1])
    })

# To run the Flask application (for local testing)
# if __name__ == '__main__':
  app.run(debug=True)
#     # You might need to change the host to '0.0.0.0' to make it accessible in Colab environments
#     app.run(host='0.0.0.0', port=5000)
