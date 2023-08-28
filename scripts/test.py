import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from utils import *
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

class ImprovedLSTMForecaster:
    def __init__(self, input_seq_len=72, output_seq_len=1, n_features=17):
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.n_features = n_features
        self.scaler = MinMaxScaler()
        self.model = None
        
    def preprocess_data(self, df, test_size=0.2):
        df['Time'] = pd.to_datetime(df['Time'])

        # Adding lags
        for i in [1, 2, 7]:
            col_name = f"Lag_{i}day"
            df[col_name] = df['Time'].apply(lambda x: df.loc[df['Time'] == (x - pd.DateOffset(days=i)), 'Load (kW)'].values[0] 
                                if (x - pd.DateOffset(days=i)) in df['Time'].values else np.nan)

        # Drop first week of data due to NaNs from lag
        cutoff = df['Time'].dt.date.min() + pd.DateOffset(days=7)
        df = df[df['Time'].dt.date >= cutoff.date()]

        # Decomposing datetime
        df['Day'] = df['Time'].dt.day
        df['Month'] = df['Time'].dt.month
        df['Hour'] = df['Time'].dt.hour
        df['Year'] = df['Time'].dt.year
        df['Weekday'] = df['Time'].dt.weekday

        # Sine/Cosine Encoding for hour and month
        df['Hour Sin'] = np.sin(df['Hour'] * 2 * np.pi / 24)
        df['Hour Cos'] = np.cos(df['Hour'] * 2 * np.pi / 24)
        df['Month Sin'] = np.sin(df['Month'] * 2 * np.pi / 12)
        df['Month Cos'] = np.cos(df['Month'] * 2 * np.pi / 12)
        df.drop(['Hour', 'Month', 'Time'], axis='columns', inplace=True)

        # Scale the data
        load_data = df['Load (kW)'].values
        features = df.drop(['Load (kW)'], axis=1).values
        scaled_features = self.scaler.fit_transform(features)

        # Concatenate Load (unscaled) with the scaled features
        data = np.column_stack((load_data, scaled_features))

        # Sequence the data for LSTM
        X, y = [], []
        for i in range(len(data) - self.input_seq_len - self.output_seq_len + 1):
            X.append(data[i:i+self.input_seq_len])
            y.append(data[i+self.input_seq_len:i+self.input_seq_len+self.output_seq_len, 0])

        X = np.array(X)
        y = np.array(y)

        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=test_size, random_state=42)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    
    def build_model(self, lstm_layers=(50,), dropout=0.2, learning_rate=0.001, activation='tanh', regularization_rate=0.01):
        model = Sequential()
        
        for i, units in enumerate(lstm_layers):
            return_seq = True if i < len(lstm_layers) - 1 else False
            if i == 0:
                model.add(LSTM(units, activation=activation, input_shape=(self.input_seq_len, self.n_features),
                               return_sequences=return_seq, kernel_regularizer=l2(regularization_rate)))
            else:
                model.add(LSTM(units, activation=activation, return_sequences=return_seq, kernel_regularizer=l2(regularization_rate)))
            model.add(Dropout(dropout))
        
        model.add(Dense(self.output_seq_len))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mape'])
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=64, patience=5):
        if self.model is None:
            print("Model not built yet. Call `build_model` first.")
            return
        
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop]
        )
        return history
    
    def predict(self, X):
        if self.model is None:
            print("Model not trained yet. Train the model first.")
            return
        y_pred = self.model.predict(X)
        # Since the Load predictions aren't scaled, we directly return them
        return y_pred[:, 0]
    
    def tune_hyperparameters(self, X, y, param_grid):
        model = KerasRegressor(build_fn=self.build_model, epochs=10, batch_size=64, verbose=1)
        
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=2)
        grid_result = grid.fit(X, y)
        
        # Return the best parameters from the grid search
        return grid_result.best_params_

df = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))

# Test the improved class
forecaster_improved = ImprovedLSTMForecaster(input_seq_len=24, output_seq_len=1, n_features=17)

# Splitting the data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = forecaster_improved.preprocess_data(df, test_size=0.2)

# Parameters for testing
param_grid_improved = {
    'lstm_layers': [(64, 128)],
    'dropout': [0.2],
    'learning_rate': [0.01],
    'activation': ['tanh'],
    'regularization_rate': [0.01]
}

# Tune hyperparameters
best_params_improved = forecaster_improved.tune_hyperparameters(X_train, y_train, param_grid_improved)

# Using the best_params, build the model
forecaster_improved.build_model(**best_params_improved)

# Train the model on the training set using the best hyperparameters
forecaster_improved.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=64, patience=5)

# Predict using the model on the test set
y_pred_improved = forecaster_improved.predict(X_test)

class LSTMForecaster:
    def __init__(self, input_seq_len=72, output_seq_len=1, n_features=7):
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.n_features = n_features
        self.scaler = MinMaxScaler()
        self.model = None
        
    def preprocess_data(self, df, test_size=0.2):
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values(by='Time')
        
        # Extract Load and features
        load_data = df['Load (kW)'].values
        features = df.drop(['Time', 'Load (kW)'], axis=1).values
        
        # Scale only the features
        scaled_features = self.scaler.fit_transform(features)
        
        # Concatenate Load (unscaled) with the scaled features
        data = np.column_stack((load_data, scaled_features))
        
        X, y = [], []
        for i in range(len(data) - self.input_seq_len - self.output_seq_len + 1):
            X.append(data[i:i+self.input_seq_len])
            y.append(data[i+self.input_seq_len:i+self.input_seq_len+self.output_seq_len, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=test_size, random_state=42)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    def build_model(self, lstm_units=50, dropout=0.2, learning_rate=0.001, activation='tanh'):
        model = Sequential()
        model.add(LSTM(lstm_units, activation=activation, input_shape=(self.input_seq_len, self.n_features), return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(lstm_units, activation=activation))
        model.add(Dropout(dropout))
        model.add(Dense(self.output_seq_len))
        model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape'])
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=64):
        if self.model is None:
            print("Model not built yet. Call `build_model` first.")
            return
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val)
        )
        return history
    
    def predict(self, X):
        if self.model is None:
            print("Model not trained yet. Train the model first.")
            return
        y_pred = self.model.predict(X)
        # Since the Load predictions aren't scaled, we directly return them
        return y_pred[:, 0]

    
    def tune_hyperparameters(self, X, y, param_grid):
        model = KerasRegressor(build_fn=self.build_model, epochs=10, batch_size=64, verbose=1)
        
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
        grid_result = grid.fit(X, y)
        
        # Return the best parameters from the grid search
        return grid_result.best_params_


# print("hello")
# # Make sure the dataset columns are in the correct order and sorted by time.
# df = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))
# print("read data")
# # Initialize the LSTMForecaster
# forecaster = LSTMForecaster(input_seq_len=72, output_seq_len=48, n_features=7)

# # Preprocess the data and split into train, validation, and test sets
# (X_train, y_train), (X_val, y_val), (X_test, y_test) = forecaster.preprocess_data(df, test_size=0.2)
# print(X_train[:1], X_test[:1])

# # Define a parameter grid for hyperparameter tuning
# param_grid = {
#     'lstm_units': [25],
#     'dropout': [0.2],
#     'learning_rate': [0.1],
#     'activation': ['tanh']
# }

# # Tune hyperparameters
# best_params = forecaster.tune_hyperparameters(X_train, y_train, param_grid)

# # Using the best_params, build the model
# forecaster.build_model(**best_params)

# # Train the model on the training set using the best hyperparameters
# forecaster.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=64)

# # Predict using the model on the test set
# y_pred = forecaster.predict(X_test)