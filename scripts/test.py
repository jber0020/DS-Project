import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from utils import *
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU


class ImprovedLSTMForecaster:
    def __init__(self, input_seq_len=72, output_seq_len=1, n_features=17):
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.n_features = n_features
        self.scaler = MinMaxScaler()
        self.model = None
        
    def preprocess_data(self, df, test_size=0.2, weather=True):
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
        if not weather:
            data = load_data.reshape(-1,1)
            self.n_features = 1

        # Sequence the data for LSTM
        X, y = [], []
        for i in range(len(data) - self.input_seq_len - self.output_seq_len + 1):
            X.append(data[i:i+self.input_seq_len])
            if weather:
                y.append(data[i+self.input_seq_len:i+self.input_seq_len+self.output_seq_len, 0])
            else:
                y.append(data[i+self.input_seq_len:i+self.input_seq_len+self.output_seq_len])

        X = np.array(X)
        y = np.array(y)

        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=test_size, random_state=42)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    
    def build_model(self, lstm_layers=(50,), dropout=0.2, learning_rate=0.001, activation='leaky_relu', regularization_rate=0.01, alpha=0.01, clip=1):
        model = Sequential()
        
        for i, units in enumerate(lstm_layers):
            return_seq = True if i < len(lstm_layers) - 1 else False
            if i == 0:
                model.add(LSTM(units, activation=None if activation == 'leaky_relu' else activation,
                            input_shape=(self.input_seq_len, self.n_features), return_sequences=return_seq, 
                            kernel_regularizer=l2(regularization_rate)))
            else:
                model.add(LSTM(units, activation=None if activation == 'leaky_relu' else activation, 
                            return_sequences=return_seq, kernel_regularizer=l2(regularization_rate)))

            if activation == 'leaky_relu':
                model.add(LeakyReLU(alpha=alpha))
            if dropout > 0:
                model.add(Dropout(dropout))
        
        model.add(Dense(self.output_seq_len))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=clip)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae', 'mape'])
        self.model = model
        return model

    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, patience=5):
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
        model = KerasRegressor(build_fn=self.build_model, epochs=10, batch_size=64, verbose=1, activation=LeakyReLU, 
                               alpha=param_grid['alpha'], dropout=param_grid['dropout'], 
                               learning_rate=param_grid['learning_rate'], lstm_layers=param_grid['lstm_layers'],
                               regularization_rate=param_grid['regularization_rate'])

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=2)
        grid_result = grid.fit(X, y)
        
        # Return the best parameters from the grid search
        return grid_result.best_params_


def LSTM_tuning(param_grid, weather=True):
    df = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))
    learning_rates = param_grid['learning_rate']
    layers = param_grid['lstm_layers']

    # Test the improved class
    forecaster_improved = ImprovedLSTMForecaster(input_seq_len=24*7, output_seq_len=48, n_features=17)

    # Splitting the data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = forecaster_improved.preprocess_data(df, test_size=0.2, weather=weather)

    results = open("LSTM_results_1week.csv", "w")
    results.write("layers,learning_rate,val_loss,mse,mae,mape\n")
    results.close()

    for lr in learning_rates:
        for l in layers:
            Model = ImprovedLSTMForecaster(input_seq_len=24*7, output_seq_len=48, n_features=1)
            Model.build_model(lstm_layers=l, learning_rate=lr, dropout=0, clip=1)
            Model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, epochs=10, batch_size=64)
            [val_loss, mse, mae, mape] = Model.model.evaluate(X_test, y_test, verbose=0)
            results = open("LSTM_results_1week.csv", "a")
            results.write("{},{},{},{},{},{}\n".format(l, lr, val_loss, mse, mae, mape))
            results.close()
            
if __name__=="__main__":
    # Parameters for testing
    param_grid = {
        'lstm_layers': [(64, 64), (128, 64), (256, 128), (512, 256), (64, 32)],
        'learning_rate': [0.0001, 0.001, 0.01],
    }
    LSTM_tuning(param_grid=param_grid, weather=False)

# df = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))

# # Test the improved class
# forecaster_improved = ImprovedLSTMForecaster(input_seq_len=24, output_seq_len=48, n_features=17)

# # Splitting the data
# (X_train, y_train), (X_val, y_val), (X_test, y_test) = forecaster_improved.preprocess_data(df, test_size=0.2)

# # Parameters for testing
# param_grid = {
#     'lstm_layers': [(64, 128)],
#     'dropout': [0.2],
#     'learning_rate': [0.0001],
#     'activation': ['leaky_relu'],
#     'alpha': [0.01, 0.05, 0.1],  # example values
#     'regularization_rate': [0.01]
# }

# # Tune hyperparameters
# # best_params = forecaster_improved.tune_hyperparameters(X_train, y_train, param_grid)

# # Using the best_params, build the model
# forecaster_improved.build_model(lstm_layers=(64, 128), dropout=0.2, learning_rate=0.001, activation='leaky_relu')

# # Train the model on the training set using the best hyperparameters
# forecaster_improved.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32, patience=5)

# # Predict using the model on the test set
# y_pred_improved = forecaster_improved.predict(X_test)
