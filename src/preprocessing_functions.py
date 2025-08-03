"""
Data preprocessing functions for agriculture yield prediction.

This module contains all the preprocessing functions extracted from the notebook
to make them testable and reusable.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import os


def load_crop_data(file_path: str) -> pd.DataFrame:
    """
    Load crop data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded crop data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise pd.errors.EmptyDataError("The CSV file is empty")
        return df
    except pd.errors.EmptyDataError as e:
        raise e  # Re-raise EmptyDataError as-is
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {str(e)}")


def filter_rice_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and extract rice-related columns from the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe with crop data
        
    Returns:
        pd.DataFrame: Filtered dataframe with only rice-related columns
        
    Raises:
        KeyError: If required columns are missing
    """
    required_columns = [
        'Year', 'State Name', 'Dist Name',
        'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)', 
        'RICE YIELD (Kg per ha)'
    ]
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    df_rice = df[required_columns].copy()
    return df_rice


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Check for missing values in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.Series: Count of missing values per column
    """
    return df.isnull().sum()


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Strategy to handle missing values ('drop', 'fill_mean', 'fill_median')
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled
        
    Raises:
        ValueError: If invalid strategy is provided
    """
    valid_strategies = ['drop', 'fill_mean', 'fill_median']
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy. Choose from: {valid_strategies}")
    
    df_processed = df.copy()
    
    if strategy == 'drop':
        df_processed = df_processed.dropna()
    elif strategy == 'fill_mean':
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(
            df_processed[numeric_cols].mean()
        )
    elif strategy == 'fill_median':
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(
            df_processed[numeric_cols].median()
        )
    
    return df_processed


def encode_categorical_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Encode a categorical column using LabelEncoder.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column_name (str): Name of the column to encode
        
    Returns:
        pd.DataFrame: Dataframe with encoded column added
        
    Raises:
        KeyError: If column doesn't exist in dataframe
    """
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in dataframe")
    
    df_encoded = df.copy()
    le = LabelEncoder()
    
    # Handle any NaN values by filling with 'Unknown'
    if df_encoded[column_name].isnull().any():
        df_encoded[column_name] = df_encoded[column_name].fillna('Unknown')
    
    encoded_column_name = f"{column_name.replace(' ', '_').lower()}_encoded"
    df_encoded[encoded_column_name] = le.fit_transform(df_encoded[column_name])
    
    return df_encoded


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> dict:
    """
    Detect outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to check for outliers
        
    Returns:
        dict: Dictionary with outlier bounds information
        
    Raises:
        KeyError: If column doesn't exist
        ValueError: If column is not numeric
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'outlier_count': len(outliers),
        'outlier_indices': outliers.index.tolist()
    }


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove outliers from dataframe using IQR method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to remove outliers from
        
    Returns:
        pd.DataFrame: Dataframe with outliers removed
    """
    outlier_info = detect_outliers_iqr(df, column)
    
    # Remove outliers
    df_clean = df[
        (df[column] >= outlier_info['lower_bound']) & 
        (df[column] <= outlier_info['upper_bound'])
    ].copy()
    
    return df_clean


def scale_features(X_train: pd.DataFrame, X_test: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (Optional[pd.DataFrame]): Test features
        
    Returns:
        Tuple: Scaled training features, scaled test features (if provided), fitted scaler
        
    Raises:
        ValueError: If training data is empty
    """
    if X_train.empty:
        raise ValueError("Training data cannot be empty")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_test_scaled = None
    if X_test is not None and not X_test.empty:
        X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of test data (0.0 to 1.0)
        random_state (int): Random state for reproducibility
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test
        
    Raises:
        ValueError: If inputs are invalid
    """
    if X.empty or y.empty:
        raise ValueError("Features and target cannot be empty")
    
    if len(X) != len(y):
        raise ValueError("Features and target must have the same length")
    
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def validate_data_format(df: pd.DataFrame) -> dict:
    """
    Validate data format and return summary statistics.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Validation summary with data quality metrics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    
    # Check for specific data quality issues
    if 'RICE YIELD (Kg per ha)' in df.columns:
        summary['negative_yield_count'] = (df['RICE YIELD (Kg per ha)'] < 0).sum()
        summary['zero_yield_count'] = (df['RICE YIELD (Kg per ha)'] == 0).sum()
    
    if 'RICE AREA (1000 ha)' in df.columns:
        summary['negative_area_count'] = (df['RICE AREA (1000 ha)'] < 0).sum()
        summary['zero_area_count'] = (df['RICE AREA (1000 ha)'] == 0).sum()
    
    if 'RICE PRODUCTION (1000 tons)' in df.columns:
        summary['negative_production_count'] = (df['RICE PRODUCTION (1000 tons)'] < 0).sum()
        summary['zero_production_count'] = (df['RICE PRODUCTION (1000 tons)'] == 0).sum()
    
    return summary


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed dataframe to CSV file.
    
    Args:
        df (pd.DataFrame): Dataframe to save
        output_path (str): Path where to save the file
        
    Raises:
        ValueError: If dataframe is empty
        PermissionError: If cannot write to the specified path
    """
    if df.empty:
        raise ValueError("Cannot save empty dataframe")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
    except PermissionError:
        raise PermissionError(f"Cannot write to {output_path}. Check file permissions.")
    except Exception as e:
        raise ValueError(f"Error saving file: {str(e)}")


def complete_preprocessing_pipeline(input_file: str, output_file: str) -> dict:
    """
    Complete preprocessing pipeline that combines all steps.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save processed data
        
    Returns:
        dict: Summary of processing steps
    """
    # Step 1: Load data
    df = load_crop_data(input_file)
    original_shape = df.shape
    
    # Step 2: Filter rice data
    df_rice = filter_rice_data(df)
    
    # Step 3: Handle missing values
    df_clean = handle_missing_values(df_rice, strategy='drop')
    
    # Step 4: Encode categorical variables
    df_encoded = encode_categorical_column(df_clean, 'State Name')
    
    # Step 5: Remove outliers from yield column
    df_final = remove_outliers_iqr(df_encoded, 'RICE YIELD (Kg per ha)')
    
    # Step 6: Validate final data
    validation_summary = validate_data_format(df_final)
    
    # Step 7: Save processed data
    save_processed_data(df_final, output_file)
    
    # Return processing summary
    summary = {
        'original_shape': original_shape,
        'final_shape': df_final.shape,
        'rows_removed': original_shape[0] - df_final.shape[0],
        'columns_added': df_final.shape[1] - df_rice.shape[1],
        'validation_summary': validation_summary,
        'output_file': output_file
    }
    
    return summary
