import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from utils import *
from kerastuner.tuners import RandomSearch

data = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))

data['Time'] = pd.to_datetime(data['Time'])

# Clean time values
data['Day'] = data['Time'].dt.day
data['Month'] = data['Time'].dt.month
data['Hour'] = data['Time'].dt.hour
data['Year'] = data['Time'].dt.year
data['Weekday'] = data['Time'].dt.weekday
data.drop("Time", axis="columns", inplace=True)

# Normalisation
normalised_columns = ~data.columns.isin(['Hour', 'Month', 'Load (kW)'])
data.loc[:,normalised_columns] = (data.loc[:,normalised_columns]-data.loc[:,normalised_columns].mean()) / data.loc[:,normalised_columns].std()

# Sine/Cosine Encoding
data['Hour Sin'] = np.sin(data['Hour'] * 2 * np.pi / 24)
data['Hour Cos'] = np.cos(data['Hour'] * 2 * np.pi / 24)
data['Month Sin'] = np.sin(data['Month'] * 2 * np.pi / 12)
data['Month Cos'] = np.cos(data['Month'] * 2 * np.pi / 12)
data.drop(['Hour', 'Month'], axis='columns', inplace=True)

# Remove unimportant columns
data.drop(['Pressure_kpa', 'Cloud Cover (%)', 'Humidity (%)'], axis='columns', inplace=True)

def Sequential_Input(df, input_sequence, output_sequence):
    df_np = df.to_numpy()
    X = []
    y = []
    
    for i in range(len(df_np) - input_sequence - output_sequence + 1):
        row = [a for a in df_np[i:i + input_sequence]]
        X.append(row)
        label = df_np[i + input_sequence:i + input_sequence + output_sequence, df.columns.get_loc("Load (kW)")]  # Extract Load (kW) values
        y.append(label)
        
    return np.array(X), np.array(y)

X, y = Sequential_Input(data, 24, 48)

n=len(X)

# Training data
X_train, y_train = X[:int(n*0.7)], y[:int(n*0.7),]

# Validation data
X_val, y_val = X[int(n*0.7):int(n*0.9)], y[int(n*0.7):int(n*0.9),]

# Test data
X_test, y_test = X[int(n*0.9):], y[int(n*0.9):,]

n_features = X_train.shape[2]                       

"""
model1 = tf.keras.Sequential()

model1.add(tf.keras.layers.InputLayer((24,n_features)))
model1.add(tf.keras.layers.LSTM(100, return_sequences = True))     
model1.add(tf.keras.layers.LSTM(100, return_sequences = False))
model1.add(tf.keras.layers.Dense(48, activation = 'linear'))

early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 2)

model1.compile(loss = 'mean_squared_error', 
               optimizer = 'Adam', 
               metrics = ['mse', 'mae', 'mape'])

model1.fit(X_train, y_train, 
           validation_data = (X_val, y_val), 
           epochs = 50, 
           callbacks = [early_stop])
"""
def build_lstm_model(hp):
    model = keras.Sequential()
    
    # Define hyperparameters
    units = hp.Int('units', min_value=32, max_value=256, step=32)
    num_layers = hp.Int('num_layers', min_value=1, max_value=3)
    
    for _ in range(num_layers-1):
        model.add(keras.layers.LSTM(units, return_sequences=True))
    model.add(keras.layers.LSTM(units, return_sequences=False))
    model.add(keras.layers.Dense(48, activation='linear'))
    
    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        metrics=['mse', 'mae', 'mape']
    )
    
    return model

tuner = RandomSearch(
    build_lstm_model,
    objective='val_loss',
    max_trials=10,  # Number of trials to search
    directory='lstm_tuning',  # Directory to store the search results
    project_name='lstm_tuning'  # Name of the tuning project
)

tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])

best_model = tuner.get_best_models(num_models=1)[0]
evaluation = best_model.evaluate(X_test, y_test)
print(evaluation)
