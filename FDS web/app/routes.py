from flask import Blueprint, render_template, request
import pickle
from .model import predict

main = Blueprint('main', __name__)

# Load the saved label encoders
with open('model/label_encoder_type.pkl', 'rb') as le_type_file:
    label_encoder_type = pickle.load(le_type_file)

with open('model/label_encoder_location.pkl', 'rb') as le_location_file:
    label_encoder_location = pickle.load(le_location_file)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def get_prediction():
    # Extract user data from the form
    amount = float(request.form['amount'])
    location = request.form['location']
    purchase_type = request.form['purchase_type']

    # Encode non-numeric inputs
    try:
        location_encoded = label_encoder_location.transform([location])[0]
    except ValueError:
        return render_template('result.html', result="Error: Unknown location entered.")

    try:
        purchase_type_encoded = label_encoder_type.transform([purchase_type])[0]
    except ValueError:
        return render_template('result.html', result="Error: Unknown purchase type entered.")

    # Prepare input for prediction (match model input order)
    input_data = [amount, location_encoded, purchase_type_encoded]

    # Log the input data for debugging
    print(f"Input Data: {input_data}")

    # Make prediction
    prediction = predict(input_data)

    # Log the raw prediction output
    print(f"Raw Prediction: {prediction}")

    # Interpret the prediction
    if prediction == 1:
        result = "Fraud"
    else:
        result = "Not Fraud"

    # Display the result on result.html
    return render_template('result.html', result=f"{result}")
