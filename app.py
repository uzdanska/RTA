import pickle
import numpy as np
from flask import Flask, request, jsonify
import random

class Perceptron():
    
    def __init__(self, learning_rate=0.01, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X, y):
        self.weights = [random.uniform(-1.0, 1.0) for _ in range(1 + X.shape[1])] 
        self.errors = []
        
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([1, -1, 1, -1])

perceptron = Perceptron()
perceptron.fit(X, y)

# Save the model to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(perceptron, model_file)

# Create a Flask application
app = Flask(__name__)

# API endpoint for GET request
@app.route('/predict_get', methods=['GET'])
def predict_get():
    try:
        sepal_length = float(request.args.get('sl'))
        petal_length = float(request.args.get('pl'))
        
        features = [sepal_length, petal_length]

        # Load the model from file
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
            
        # Make prediction using the model
        predicted_class = int(model.predict(features))
        
        # Return JSON response
        return jsonify(features=features, predicted_class=predicted_class)
    except Exception as e:
        return jsonify(error=str(e))

# API endpoint for POST request
@app.route('/predict_post', methods=['POST'])
def predict_post():
    try:
        data = request.get_json(force=True)
        sepal_length = float(data.get('sl'))
        petal_length = float(data.get('pl'))
        
        features = [sepal_length, petal_length]

        # Load the model from file
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
            
        # Make prediction using the model
        predicted_class = int(model.predict(features))
        
        # Return JSON response
        return jsonify(features=features, predicted_class=predicted_class)
    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    app.run()