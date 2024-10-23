from flask import Blueprint, render_template, request
from .model import predict

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def get_prediction():
    # Extract user data from the form
    amount = float(request.form['amount'])
    location = request.form['location']
    purchase_type = request.form['purchase_type']

    # Example input format for prediction (adjust based on your model's input)
    input_data = [float(amount), location, purchase_type]
    
    result = predict(input_data)
    
    return render_template('result.html', result=result)
