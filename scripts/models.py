"""
MODELLING FILE

FIT3164
Project 4 - Electricity Demand Forecasting
Team 4
- Joshua Berg
- Ryan Hendler
- Yuechuan Li
- Yangyi Li

This file contains the code for the models, feature selection, and data preprocessing.
"""

# Imports
from utils import *
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor

def feature_selection(data:pd.DataFrame) -> pd.Series:
    # Seperate Data into Dependent and Independent Variables
    x = data.drop('load_kw', axis=1)
    y = data['load_kw']
    # Normalise Data
    x = (x-x.mean())/x.std()
    y = (y-y.mean())/y.std()
    # Calculate Mutual Information Regression
    mir_scores = mutual_info_regression(x, y)
    mir_scores = pd.Series(mir_scores)
    mir_scores.index = x.columns
    return mir_scores.sort_values(ascending=False)

def preprocessing(data:pd.DataFrame) -> pd.DataFrame:
    data['time'] = pd.to_datetime(data['time'])

    # Add Lag Values
    data['lag_1day'] = data['time'].apply(lambda x: data.loc[data['time'] == (x - pd.DateOffset(days=1)), 
                                                            'load_kw'].values[0] if (x - pd.DateOffset(days=1)) in data['time'].values else None)
    data['lag_2day'] = data['time'].apply(lambda x: data.loc[data['time'] == (x - pd.DateOffset(days=2)), 
                                                            'load_kw'].values[0] if (x - pd.DateOffset(days=2)) in data['time'].values else None)
    data['lag_1week'] = data['time'].apply(lambda x: data.loc[data['time'] == (x - pd.DateOffset(days=7)), 
                                                            'load_kw'].values[0] if (x - pd.DateOffset(days=7)) in data['time'].values else None)
    
    # Remove Rows with no lags
    data.dropna(inplace=True)

    # Clean time values
    data['day'] = data['time'].dt.day
    data['month'] = data['time'].dt.month
    data['hour'] = data['time'].dt.hour
    data['year'] = data['time'].dt.year
    data['weekday'] = data['time'].dt.weekday
    data.drop("time", axis="columns", inplace=True)

    # Normalisation
    normalised_columns = ~data.columns.isin(['hour', 'month',"load_kw"])
    data.loc[:,normalised_columns] = (data.loc[:,normalised_columns]-data.loc[:,normalised_columns].mean()) / data.loc[:,normalised_columns].std()

    # Sine/Cosine Encoding
    data['hour_sin'] = np.sin(data['hour'] * 2 * np.pi / 24)
    data['hour_cos'] = np.cos(data['hour'] * 2 * np.pi / 24)
    data['month_sin'] = np.sin(data['month'] * 2 * np.pi / 12)
    data['month_cos'] = np.cos(data['month'] * 2 * np.pi / 12)
    data.drop(['hour', 'month'], axis='columns', inplace=True)
        
    return data

class xgBoostForecaster:
    def __init__(self) -> None:
        pass

    def build_model(self, n_estimators=100, max_depth=3, learning_rate=0.1) -> None:
        # Set parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        # Build model
        self.model = XGBRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate)

    def train_model(self, X_train, y_train, X_val, y_val):
        # Train model
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=True, eval_metric='mse')

    def test_model(self, X_test, y_test):
        pass

    def predict(self, x):
        # Return prediction
        return self.model.predict(x)

    def save_model(self) -> None:
        pass

    def load_model(self) -> None:
        pass

class NeuralNetForecaster:
    def __init__(self) -> None:
        self.model = None
    
    def build_model(self, input_size, layers, batch_size=32, learning_rate=0.001, epochs=100, metrics=['mse', 'mae', 'mape']):
        # Set Parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.input_size = input_size

        # Build Model
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(layers[0], input_dim=input_size))
        if len(layers) > 1:
            for i in layers[1:]:
                self.model.add(tf.keras.layers.Dense(i, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation='relu'))
        self.model.compile(loss='mean_squared_error', metrics=self.metrics)

    def train_model(self, X_train, y_train, X_val, y_val):
        # Check model has been built
        if self.model is None:
            print('Must build model first! Call NeuralNetForecaster.build_model()')
            return
        # Train model
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.epochs, batch_size=self.batch_size)
        return history

    def test_model(self, X_test, y_test):
        # Test model
        results = self.model.evaluate(X_test, y_test, verbose=0)
        print('Validation Loss:', results[0])
        for i in range(len(self.metrics)):
            print(self.metrics[i] + ": " + str(results[i+1]))
        return results
    
    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, name, filepath):
        # Save model
        self.model.save('{}/{}.keras'.format(filepath, name))

    def load_model(self, name, filepath):
        # Load model
        self.model = tf.keras.models.load_model('{}/{}.keras'.format(filepath, name))

class RandomForestForecaster:
    def __init__(self):
        pass

