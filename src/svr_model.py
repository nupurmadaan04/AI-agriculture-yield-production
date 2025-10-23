import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle


class SVRModel:
    """Support Vector Regression model with multiple kernel options."""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', epsilon=0.1, degree=3):
        """Initialize SVR model with specified parameters."""
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.degree = degree
        self.model = None
        

    def train(self, X_train, y_train):
        """Train the SVR model."""
        self.model = SVR(
            kernel=self.kernel, 
            C=self.C, 
            gamma=self.gamma, 
            epsilon=self.epsilon, 
            degree=self.degree
        )
        self.model.fit(X_train, y_train)
        

    def predict(self, X_test):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X_test)


    def model_evaluation(self, y_test, y_pred):
        """Evaluate the model and return metrics."""
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Print for display
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        
        # Return values for programmatic use
        return mse, r2, mae


    def save_model(self, filepath):
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    @staticmethod
    def load_model(filepath):
        """Load trained model from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)


    def __repr__(self):
        """String representation of the model."""
        trained = "trained" if self.model is not None else "not trained"
        return f"SVRModel(kernel='{self.kernel}', C={self.C}, gamma='{self.gamma}', epsilon={self.epsilon}, {trained})"

    def plot_learning_curve(self, X, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=5):
        """Plot learning curve for the model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y, train_sizes=train_sizes, cv=cv, 
            scoring='neg_mean_squared_error')

        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, label='Training score', marker='o')
        plt.plot(train_sizes, test_scores_mean, label='Cross-validation score', marker='s')
        plt.title(f'Learning Curve - {self.kernel} kernel')
        plt.xlabel('Training examples')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    print("="*60)
    print("SVR Model Training for Rice Yield Prediction")
    print("="*60)
    
    # Load Dataset
    print("\n1. Loading data...")
    df = pd.read_csv('E:/PROGRAMMING-2/AI-agriculture-yield-production/Datasets/Crops_data.csv')
    print(f"   Original data shape: {df.shape}")
    print(f"   Columns: {list(df.columns[:10])}...")  # Show first 10 columns
    
    # Check if rice columns exist
    print("\n2. Checking rice-related columns...")
    rice_area_col = 'RICE AREA (1000 ha)'
    rice_prod_col = 'RICE PRODUCTION (1000 tons)'
    rice_yield_col = 'RICE YIELD (Kg per ha)'
    
    if rice_area_col not in df.columns:
        print(f"   ERROR: Column '{rice_area_col}' not found!")
        print(f"   Available columns: {df.columns.tolist()}")
        exit(1)
    
    print(f"   ✓ Found: {rice_area_col}")
    print(f"   ✓ Found: {rice_prod_col}")
    print(f"   ✓ Found: {rice_yield_col}")
    
    # Filter rice data - keep only necessary columns
    print("\n3. Filtering rice data...")
    rice_cols = ['Year', 'State Name', 'Dist Name', 
                 rice_area_col, rice_prod_col, rice_yield_col]
    df_rice = df[rice_cols].copy()
    print(f"   Rice data shape: {df_rice.shape}")
    
    # Handle missing values
    print("\n4. Handling missing values...")
    print(f"   Missing values before: {df_rice.isnull().sum().sum()}")
    df_clean = df_rice.dropna()
    print(f"   After removing NaN: {df_clean.shape}")
    
    # Encode State Name
    print("\n5. Encoding categorical variables...")
    le = LabelEncoder()
    df_clean['State_Encoded'] = le.fit_transform(df_clean['State Name'])
    print(f"   Encoded {len(le.classes_)} unique states")
    print(f"   States: {le.classes_[:5]}...")  # Show first 5 states
    
    # Remove outliers using IQR
    print("\n6. Removing outliers from RICE YIELD...")
    Q1 = df_clean[rice_yield_col].quantile(0.25)
    Q3 = df_clean[rice_yield_col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    print(f"   IQR bounds: [{lower:.2f}, {upper:.2f}]")
    
    df_final = df_clean[(df_clean[rice_yield_col] >= lower) & 
                         (df_clean[rice_yield_col] <= upper)]
    print(f"   After outlier removal: {df_final.shape}")
    print(f"   Removed {len(df_clean) - len(df_final)} outliers")
    
    # Prepare features and target
    print("\n7. Preparing features and target...")
    feature_cols = ['Year', rice_area_col, rice_prod_col, 'State_Encoded']
    X = df_final[feature_cols]
    y = df_final[rice_yield_col]
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   Feature columns: {feature_cols}")
    
    # Split data
    print("\n8. Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train size: {X_train.shape[0]} samples")
    print(f"   Test size: {X_test.shape[0]} samples")
    
    # Scale features
    print("\n9. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"   ✓ Features scaled using StandardScaler")
    
    # Train SVR models with different kernels
    print("\n10. Training SVR models with different kernels...")
    kernels = ['rbf', 'poly', 'linear']
    results = {}
    
    for kernel in kernels:
        print(f"\n{'='*60}")
        print(f"Training SVR with {kernel.upper()} kernel...")
        print('='*60)
        
        # Create and train model
        svr = SVRModel(kernel=kernel)
        print(f"Model: {svr}")
        
        print("Training...")
        svr.train(X_train_scaled, y_train)
        print(f"✓ Training complete")
        
        # Predict
        print("Making predictions...")
        y_pred = svr.predict(X_test_scaled)
        
        # Evaluate
        print("Evaluation metrics:")
        mse, r2, mae = svr.model_evaluation(y_test, y_pred)
        results[kernel] = {'MSE': mse, 'R2': r2, 'MAE': mae}
        
        # Save model
        model_path = f'../Models/svr_{kernel}_model.pkl'
        svr.save_model(model_path)
        print(f"✓ Model saved to {model_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - Model Comparison")
    print('='*60)
    print(f"{'Kernel':<10} {'MSE':<15} {'R²':<15} {'MAE':<15}")
    print('-'*60)
    for kernel, metrics in results.items():
        print(f"{kernel.upper():<10} {metrics['MSE']:<15.2f} {metrics['R2']:<15.4f} {metrics['MAE']:<15.2f}")
    
    # Find best model
    best_kernel = max(results, key=lambda k: results[k]['R2'])
    print(f"\n✓ Best model: {best_kernel.upper()} (R² = {results[best_kernel]['R2']:.4f})")
    print(f"\n✓ All models trained and saved successfully!")