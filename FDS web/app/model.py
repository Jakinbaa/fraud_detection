import pickle

# Load the pre-trained model
def load_model():
    with open('model/model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def predict(data):
    model = load_model()
    return model.predict([data])[0]
