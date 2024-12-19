import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle


def predict(input_data):
    # Load the trained model
    with open('model/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Feature names used during training
    feature_names = ['type_encoded', 'amount', 'location_encoded']

    # Ensure the input data is in the correct format with feature names
    if isinstance(input_data, list):
        input_data = pd.DataFrame([input_data], columns=feature_names)

    # Log the input data for debugging
    print(f"Input Data for Prediction:\n{input_data}")

    # Make prediction
    return model.predict(input_data)[0]


if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv('data/FDS.csv')

    # Encode categorical variables
    label_encoder_type = LabelEncoder()
    label_encoder_location = LabelEncoder()

    data['type_encoded'] = label_encoder_type.fit_transform(data['type'])
    data['location_encoded'] = label_encoder_location.fit_transform(data['location'])

    # Sample 10% of the data (optional, can be adjusted based on dataset size)
    data_sample = data.sample(frac=0.1, random_state=42)

    # Features and target variable
    X_sample = data_sample[['type_encoded', 'amount', 'location_encoded']]
    y_sample = data_sample['is_fraud']

    # Balance the dataset using SMOTE
    smote = SMOTE(random_state=42)
    X_sample, y_sample = smote.fit_resample(X_sample, y_sample)

    # Debug: Check class distribution after SMOTE
    print(f"Class Distribution After SMOTE: {y_sample.value_counts()}")

    # Split data into training and testing sets
    X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_sample, y_train_sample)

    # Save the trained model
    with open('model/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    # Save the label encoders
    with open('model/label_encoder_type.pkl', 'wb') as le_type_file:
        pickle.dump(label_encoder_type, le_type_file)

    with open('model/label_encoder_location.pkl', 'wb') as le_location_file:
        pickle.dump(label_encoder_location, le_location_file)

    # Evaluate the model
    y_pred_sample = model.predict(X_test_sample)
    accuracy_sample = accuracy_score(y_test_sample, y_pred_sample)

    # Debug: Check the distribution of predictions
    fraud_count = sum(y_pred_sample)
    not_fraud_count = len(y_pred_sample) - fraud_count
    print(f"Fraud Predictions: {fraud_count}")
    print(f"Not Fraud Predictions: {not_fraud_count}")

    print(f"Accuracy: {accuracy_sample * 100:.2f}%")

    # Debug: Feature importance
    importances = model.feature_importances_
    feature_names = ['type_encoded', 'amount', 'location_encoded']
    print("Feature Importance:")
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.4f}")

    # Test predictions with sample inputs
    print("Testing sample predictions...")

    # Example: large amount, high-risk location, online transaction
    test_input_fraud = [1, 1000.0, 2]

    # Example: small amount, low-risk location, in-store transaction
    test_input_not_fraud = [0, 20.0, 1]

    print(f"Fraud Test Prediction: {predict(test_input_fraud)}")
    print(f"Not Fraud Test Prediction: {predict(test_input_not_fraud)}")

    # Test prediction with a training sample
    sample_train_data = X_train_sample.iloc[0].tolist()
    print(f"Training Sample Prediction: {predict(sample_train_data)}")
    print(f"Actual Label: {y_train_sample.iloc[0]}")
