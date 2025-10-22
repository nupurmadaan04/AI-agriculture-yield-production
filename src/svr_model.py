import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.SVM import SVR
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv('E:/PROGRAMMING-2/AI-agriculture-yield-production/Datasets/Crops_data.csv')