import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pickle

# Load Dataset
df = pd.read_csv('AI-agriculture-yield-production/Datasets/Crops_data.csv')


class SVRModel:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', epsilon=0.1, degree=3):
        # Initialize SVR model with specified parameters
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.degree = degree
        self.model = None
        

    def train(self, X_train, y_train):
        # Train the SVR model
        self.model = SVR(
            kernel=self.kernel, 
            C=self.C, 
            gamma=self.gamma, 
            epsilon=self.epsilon, 
            degree=self.degree
        )
        self.model.fit(X_train, y_train)
        

    def predict(self, X_test):
        # Make predictions
        if self.model is not None:
            return self.model.predict(X_test)
        else:
            raise ValueError("Model not trained.")


    def model_evaluation(self, y_test, y_pred):
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return f"Mean Squared Error: {mse}, \nR2 Score: {r2}, \nMean Absolute Error: {mae}"


    def save_model(self, filepath):
        # Save trained model to file
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        dir_path = os.path.dirname(filepath) # Create directory if not existing
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)


    def __repr__(self):
        return f"SVRModel(kernel='{self.kernel}', C={self.C}, gamma='{self.gamma}', epsilon={self.epsilon}, {trained})"
