import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
data = pd.read_csv('data/FDS.csv')

# Encode categorical variables
label_encoder_type = LabelEncoder()
label_encoder_location = LabelEncoder()

data['type_encoded'] = label_encoder_type.fit_transform(data['type'])
data['location_encoded'] = label_encoder_location.fit_transform(data['location'])

# Sample 10% of the data
data_sample = data.sample(frac=0.1, random_state=42)

# Features and target variable
X_sample = data_sample[['type_encoded', 'amount', 'location_encoded']]
y_sample = data_sample['is_fraud']

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

print(f"Accuracy: {accuracy_sample * 100:.2f}%")
