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
from sklearn.ensemble import RandomForestRegressor
import math

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

def preprocessing(df:pd.DataFrame, initial=False) -> pd.DataFrame:
    data = df[['time', 'load_kw', 'pressure_kpa', 'cloud_cover_pct', 'temperature_c',
                     'wind_direction_deg', 'wind_speed_kmh']]
    data = data.copy()
    data['time'] = pd.to_datetime(data['time'], format='%d/%m/%Y %H:%M')

    # Add Lag Values
    data['lag_1day'] = data['time'].apply(lambda x: data.loc[data['time'] == (x - pd.DateOffset(days=1)), 
                                                            'load_kw'].values[0] if (x - pd.DateOffset(days=1)) in data['time'].values else None)
    data['lag_2day'] = data['time'].apply(lambda x: data.loc[data['time'] == (x - pd.DateOffset(days=2)), 
                                                            'load_kw'].values[0] if (x - pd.DateOffset(days=2)) in data['time'].values else None)
    data['lag_1week'] = data['time'].apply(lambda x: data.loc[data['time'] == (x - pd.DateOffset(days=7)), 
                                                            'load_kw'].values[0] if (x - pd.DateOffset(days=7)) in data['time'].values else None)
    
    # Remove rows with no lag
    data.dropna(subset=['lag_1week'], inplace=True)

    # Clean time values
    data['day'] = data['time'].dt.day
    data['month'] = data['time'].dt.month
    data['hour'] = data['time'].dt.hour
    data['year'] = data['time'].dt.year
    data['weekday'] = data['time'].dt.weekday
    data.drop("time", axis="columns", inplace=True)
    # Normalisation
    normalised_columns = ~data.columns.isin(['hour', 'month',"load_kw"])
    if initial:
        data.loc[:,normalised_columns].mean().to_csv(get_root("scripts/means.csv"))
        data.loc[:,normalised_columns].std().to_csv(get_root('scripts/stds.csv'))
        data.loc[:,normalised_columns] = (data.loc[:,normalised_columns]-data.loc[:,normalised_columns].mean()) / data.loc[:,normalised_columns].std()
    else:
        means = pd.read_csv(get_root('scripts/means.csv'), index_col=0)
        means = means.astype(float)
        stds = pd.read_csv(get_root('scripts/stds.csv'), index_col=0)
        stds = stds.astype(float)
        data.loc[:,normalised_columns] = (data.loc[:,normalised_columns]-means.iloc[:,0]) / stds.iloc[:,0]
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

    def train_model(self, X_train, y_train):
        # Train model
        self.model.fit(X_train, y_train, verbose=True)

    def test_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        print('MSE: {}'.format(mse))
        print('MAE: {}'.format(mae))
        print('MAPE: {}'.format(mape*100))
        return [mse, mae, mape*100]

    def predict(self, x):
        # Return prediction
        return self.model.predict(x)

    def save_model(self, name, filepath):
        self.model.save_model(get_root('{}/{}.model'.format(filepath, name)))

    def load_model(self, name, filepath):
        self.build_model()
        self.model.load_model(get_root('{}/{}.model'.format(filepath, name)))

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
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=self.metrics)

    def train_model(self, X_train, y_train, X_val, y_val):
        # Check model has been built
        if self.model is None:
            print('Must build model first! Call NeuralNetForecaster.build_model()')
            return
        # Train model
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                                 epochs=self.epochs, batch_size=self.batch_size,
                                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                                             patience=10)],
                                                                             verbose=0)
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
        self.model.save(get_root('{}/{}.keras'.format(filepath, name)))

    def load_model(self, name, filepath):
        # Load model
        self.model = tf.keras.models.load_model(get_root('{}/{}.keras'.format(filepath, name)))

class RandomForestForecaster:
    def __init__(self):
        pass

    def build_model(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=52)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def test_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        print('MSE: {}'.format(mse))
        print('MAE: {}'.format(mae))
        print('MAPE: {}'.format(mape*100))
        return [mse, mae, mape*100]

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, name, filepath):
        self.model.save_model(get_root('{}/{}.model'.format(filepath, name)))

    def load_model(self, name, filepath):
        self.build_model()
        self.model.load_model(get_root('{}/{}.model'.format(filepath, name)))


def naive(y, i:int):
    pred = y.shift(i)
    mse = mean_squared_error(y[i:], pred[i:])
    mae = mean_absolute_error(y[i:], pred[i:])
    mape = mean_absolute_percentage_error(y[i:], pred[i:])*100
    return [mse, mae, mape]


def get_forecasts(data):
    """
    Pass this in the data from the previous week plus the two days to forecast, 
    and it will return the forecasts for the next two days.

    For example, if we wish to forecast Tuesday and Wednesday, pass in data
    from last week's Tuesday to this week's Monday, plus the forecasted weather
    variables for Tuesday and Wednesday. 
    """
    # Preprocess Data
    processed_data = preprocessing(data).tail(48)

    # Break data into two days
    day1 = processed_data[:24].drop('load_kw', axis='columns')
    day2 = processed_data[24:].drop('load_kw', axis='columns')
    day2.drop('lag_1day', axis='columns', inplace=True)
    # Forecast first day
    Xgb1 = xgBoostForecaster()
    Xgb1.load_model('xgb1', 'scripts/models')
    day_one_forecasts = pd.DataFrame(Xgb1.predict(day1))

    # Forecast second day
    Xgb2 = xgBoostForecaster()
    Xgb2.load_model('xgb2', 'scripts/models')
    day_two_forecasts = pd.DataFrame(Xgb2.predict(day2))

    # Join forecasts
    joint = pd.concat([day_one_forecasts, day_two_forecasts], ignore_index=True)
    joint.columns = ['forecasts']
    joint['time'] = data.tail(48)['time'].reset_index(drop=True)

    # Return results
    return joint


def retraining_required(actuals, forecasts) -> bool:
    """
    Input last 3 weeks of actuals, 2 weeks of forecasts.
    This function will compare the performance of the 1
    week naive model against our model, over the previous
    2 weeks.
    """
    y = actuals['load_kw']
    naive_mse = naive(y, 24*7*2)[0]
    model_mse = mean_squared_error(y.tail(24*7*2), forecasts['forecasts'])
    if model_mse > naive_mse:
        return True
    else:
        return False

def retrain_model(data) -> None:
    data = preprocessing(data, initial=True)
    
    X = data.drop('load_kw', axis='columns', inplace=False)
    y = data['load_kw']

    # Calculate Number of Days
    n = len(X) // 24
    # Approx 70% train, 20% validation, 10% test
    X1_train = X[:math.floor(n*0.7)*24]
    X1_val = X[math.floor(n*0.7)*24:math.floor(n*0.9)*24]
    X1_test = X[math.floor(n*0.9)*24:]

    y_train = y[:math.floor(n*0.7)*24]
    y_val = y[math.floor(n*0.7)*24:math.floor(n*0.9)*24]
    y_test = y[math.floor(n*0.9)*24:]

    X2_train = X1_train.drop('lag_1day', axis="columns", inplace=False)
    X2_val = X1_val.drop('lag_1day', axis="columns", inplace=False)
    X2_test = X1_test.drop('lag_1day', axis="columns", inplace=False)
    
    # Run XGBoost Model 1
    XGBoost = xgBoostForecaster()
    XGBoost.build_model()
    XGBoost.train_model(X1_train, y_train)
    XGBoost.save_model('xgb1', 'scripts/models')

    # Run XGBoost Model 2
    XGBoost = xgBoostForecaster()
    XGBoost.build_model()
    XGBoost.train_model(X2_train, y_train)
    XGBoost.save_model('xgb2', 'scripts/models')

def get_insights(forecasts, actuals):
    """

    Possible insights:
    - Forecasted Peak Demand
    - Forecasted Average Demand
    - Forecasted Minimum
    - Model Performance (MAPE)

    """
    pass

if __name__=='__main__':
    data = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))
    # Grab Last Week + 2 Days to forecast
    data = data.tail(14*24)
    forecasts = get_forecasts(data)

    # Create a line plot
    plt.figure(figsize=(10, 6))
    plt.plot(forecasts['time'], forecasts['forecasts'], marker='o', linestyle='-', color='blue')
    plt.plot(data.tail(48)['time'], data.tail(48)['load_kw'], marker='o', linestyle='-', color='orange')

    # Add labels and a title
    plt.xlabel('Time')
    plt.ylabel('Forecasts')
    plt.title('Forecasts Over Time')

    plt.xticks(rotation=45)

    plt.legend(['Forecasts', "Actuals"])

    # Display the plot
    plt.grid(True)
    plt.tight_layout() 
    plt.show()

    # Test Retraining
    data = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))
    data = data.tail(24*7*3) # Grab last 3 weeks
    forecasts = pd.DataFrame()
    forecasts['forecasts'] = np.zeros(24*7*2)
    print(retraining_required(data, forecasts))
