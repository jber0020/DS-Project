# Imports
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import *
from wrangling import *
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA as ARIMA
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


class XGBoostForecaster:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, X_train, y_train):
        self.model = XGBRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        if self.model is None:
            print("Model not trained yet. Call `fit` method first.")
            return None

        y_pred = self.model.predict(X)
        return y_pred


class ARIMAForecaster:
    def __init__(self, p, d, q):
        self.p = p
        self.d = d
        self.q = q
        self.model = None

    def fit(self, train_data):
        self.model = ARIMA(train_data, order=(self.p, self.d, self.q))
        self.model_fit = self.model.fit()

    def predict(self, steps):
        if self.model is None:
            print("Model not trained yet. Call `fit` method first.")
            return None

        forecast = self.model_fit.forecast(steps=steps)
        return forecast

    def evaluate(self, test_data, forecast_steps):
        mse_scores = []
        mae_scores = []
        mape_scores = []

        num_intervals = len(test_data) // forecast_steps

        for i in range(num_intervals):
            start_idx = i * forecast_steps
            end_idx = (i + 1) * forecast_steps

            interval_test_data = test_data.iloc[start_idx:end_idx]

            forecast = self.model_fit.forecast(steps=forecast_steps)
            true_values = interval_test_data

            mse_scores.append(mean_squared_error(true_values, forecast))
            mae_scores.append(mean_absolute_error(true_values, forecast))
            mape_scores.append(mean_absolute_percentage_error(true_values, forecast))

        avg_mse = np.mean(mse_scores)
        avg_mae = np.mean(mae_scores)
        avg_mape = np.mean(mape_scores) * 100

        return avg_mse, avg_mae, avg_mape

    def plot_forecast(self, train_data, test_data, forecast, date_range):
        plt.figure(figsize=(12, 6))
        plt.plot(train_data.index, train_data, label='Train Data')
        plt.plot(test_data.index, test_data, label='Test Data')
        plt.plot(date_range, forecast, label='ARIMA Forecast', color='red')

        plt.xlabel('Time')
        plt.ylabel('Load (kW)')
        plt.title('ARIMA Forecasting')
        plt.legend()
        plt.show()


class ANN:
    """
    ANN Model
    """

    def __init__(self, input_size, layers, batch_size=32, learning_rate=0.001, epochs=100, metrics=['mse']) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.metrics= metrics
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(layers[0], input_dim=input_size, activation='relu'))
        if len(layers) > 1:
            for i in layers[1:]:
                self.model.add(tf.keras.layers.Dense(i, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=self.metrics)

    def ANN_clean_data(data):
        """
        This function prepares the data into a format for the ANN model
        """
        data['Time'] = pd.to_datetime(data['Time'])

        # Add previous day's value
        data['Lag_1day'] = data['Time'].apply(lambda x: data.loc[data['Time'] == (x - pd.DateOffset(days=1)), 
                                                            'Load (kW)'].values[0] if (x - pd.DateOffset(days=1)) in data['Time'].values else None)
        data['Lag_2day'] = data['Time'].apply(lambda x: data.loc[data['Time'] == (x - pd.DateOffset(days=2)), 
                                                            'Load (kW)'].values[0] if (x - pd.DateOffset(days=2)) in data['Time'].values else None)
        data['Lag_1week'] = data['Time'].apply(lambda x: data.loc[data['Time'] == (x - pd.DateOffset(days=7)), 
                                                            'Load (kW)'].values[0] if (x - pd.DateOffset(days=7)) in data['Time'].values else None)
        # Remove first week
        cutoff = data['Time'].dt.date.min() + pd.DateOffset(days=7)
        data = data[data['Time'].dt.date >= cutoff.date()]

        # Clean time values
        data['Day'] = data['Time'].dt.day
        data['Month'] = data['Time'].dt.month
        data['Hour'] = data['Time'].dt.hour
        data['Year'] = data['Time'].dt.year
        data['Weekday'] = data['Time'].dt.weekday
        data.drop("Time", axis="columns", inplace=True)

        # Normalisation
        normalised_columns = ~data.columns.isin(['Hour', 'Month',"Load (kW)"])
        data.loc[:,normalised_columns] = (data.loc[:,normalised_columns]-data.loc[:,normalised_columns].mean()) / data.loc[:,normalised_columns].std()

        # Sine/Cosine Encoding
        data['Hour Sin'] = np.sin(data['Hour'] * 2 * np.pi / 24)
        data['Hour Cos'] = np.cos(data['Hour'] * 2 * np.pi / 24)
        data['Month Sin'] = np.sin(data['Month'] * 2 * np.pi / 12)
        data['Month Cos'] = np.cos(data['Month'] * 2 * np.pi / 12)
        data.drop(['Hour', 'Month'], axis='columns', inplace=True)
        
        return data

    def train_model(self, X_train, y_train):
        history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)
        return self.model, history

    def test_model(self, X_val, y_val):
        results = self.model.evaluate(X_val, y_val, verbose=0)
        print('Validation Loss:', results[0])
        for i in range(len(self.metrics)):
            print(self.metrics[i] + ": " + str(results[i+1]))
        return results

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self):
        pass

    def predict(self):
        pass


def run_benchmarks():
    data = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))

    X, y = data.drop("Load (kW)", axis="columns"), data["Load (kW)"]

    # Naive
    pred = y.shift(1)
    print("===1h Naive Model===")
    print("MSE:", mean_squared_error(y[1:], pred[1:]))
    print("MAE:", mean_absolute_error(y[1:], pred[1:]))
    print("MAPE:", mean_absolute_percentage_error(y[1:], pred[1:])*100)

    # Seasonal Naive - Day
    pred = y.shift(24)
    print("===24h Naive Model===")
    print("MSE:", mean_squared_error(y[24:], pred[24:]))
    print("MAE:", mean_absolute_error(y[24:], pred[24:]))
    print("MAPE:", mean_absolute_percentage_error(y[24:], pred[24:])*100)

    # Seasonal Naive - 2 Day
    pred = y.shift(48)
    print("===48h Naive Model===")
    print("MSE:", mean_squared_error(y[48:], pred[48:]))
    print("MAE:", mean_absolute_error(y[48:], pred[48:]))
    print("MAPE:", mean_absolute_percentage_error(y[48:], pred[48:])*100)

    # Seasonal Naive - Week
    pred = y.shift(24*7)
    print("===1week Naive Model===")
    print("MSE:", mean_squared_error(y[24*7:], pred[24*7:]))
    print("MAE:", mean_absolute_error(y[24*7:], pred[24*7:]))
    print("MAPE:", mean_absolute_percentage_error(y[24*7:], pred[24*7:])*100)

    # Seasonal Naive - Month
    pred = y.shift(24*30)
    print("===1month Naive Model===")
    print("MSE:", mean_squared_error(y[24*30:], pred[24*30:]))
    print("MAE:", mean_absolute_error(y[24*30:], pred[24*30:]))
    print("MAPE:", mean_absolute_percentage_error(y[24*30:], pred[24*30:])*100)

    # Seasonal Naive - Year
    pred = y.shift(24*365)
    print("===1year Naive Model===")
    print("MSE:", mean_squared_error(y[24*365:], pred[24*365:]))
    print("MAE:", mean_absolute_error(y[24*365:], pred[24*365:]))
    print("MAPE:", mean_absolute_percentage_error(y[24*365:], pred[24*365:])*100)

    # Random Walk with Drift
    np.random.seed(52)
    n_steps = 48
    drift = 1
    sigma = y.std()
    mapes = []
    maes = []
    mses = []

    # Define Random Walk with Drift
    def random_walk(drift):
        for i in range(len(y)-48):
            walk = [y[i]]
            for _ in range(n_steps):
                walk.append(walk[-1] + drift + np.random.normal(scale=sigma))
            actuals = y[i+1:i+49]
            mses.append(mean_squared_error(actuals, walk[1:]))
            maes.append(mean_absolute_error(actuals, walk[1:]))
            mapes.append(mean_absolute_percentage_error(actuals, walk[1:])*100)
        return np.mean(mses)
    
    # Optimise Drift (Uncomment to Run, takes ages to optimise)
    # drift = minimize(random_walk, drift).x[0]
    # print("Best Drift:", drift)

    mapes = []
    maes = []
    mses = []

    # Run Optimised Random Walk
    random_walk(drift)
    print("===Random Walk with Drift===")
    print("MSE:", np.mean(mses))
    print("MAE:", np.mean(maes))
    print("MAPE:", np.mean(mapes))



def ANN_tuning():
    data = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))

    data = ANN.ANN_clean_data(data)

    X, y = data.drop("Load (kW)", axis="columns"), data["Load (kW)"]

    # Remove Unneeded Columns
    X.drop(['Day', 'Pressure_kpa', 'Cloud Cover (%)', 'Humidity (%)'], axis=1, inplace=True)

    # Optional Drop 1 Day Lag, Uncomment to Include
    # X.drop(['Lag_1day'], axis='columns', inplace=True)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    layers = [
        [64, 32, 16],
        [64, 32],
        [32, 16, 8],
        [32, 16],
        [128, 64, 32]
    ]

    for lr in learning_rates:
        for l in layers:
            Model = ANN(input_size = len(X.columns), layers = l, metrics=['mse', 'mae', 'mape'], learning_rate=lr)
            model, history = Model.train_model(X_train, y_train)
            val_loss, mse, mae, mape = Model.test_model(X_val, y_val)
            results = open("results.txt", 'a')
            results.write("{},{},{},{},{},{},{}\n".format(lr, l, val_loss, mse, mae, mape, "yes"))
            results.close()


def run_ARIMA():
    # Load your dataset
    print("Reading Data...")
    df = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))
    df['Time'] = pd.to_datetime(df['Time'])

    # Split the data into train and test sets (using 80% for training and 20% for testing)
    print("Splitting Data...")
    split_index = int(0.8 * len(df))
    train_data = df['Load (kW)'].iloc[:split_index]
    test_data = df['Load (kW)'].iloc[split_index:]

    # Define the forecasting period (48 hours)
    forecast_steps = 48

    input_sizes = [5, 12, 24, 48, 72, 24*7]

    results = open("ARIMA_results.csv", "w")
    results.write("input_size,mse,mae,mape\n")
    results.close()

    for i in input_sizes:
        # Set up the forecaster
        print("===== Input Size {} =====".format(i))
        print("Creating ARIMA...")
        forecaster = ARIMAForecaster(p=i, d=1, q=0)  # Using a week's worth of lagged inputs

        # Train the model on the entire dataset
        print("Training Model...")
        forecaster.fit(train_data)

        # Evaluate the model using 2-day intervals
        print("Evaluating Model...")
        avg_mse, avg_mae, avg_mape = forecaster.evaluate(test_data, forecast_steps)

        print(f"Average MSE: {avg_mse:.2f}")
        print(f"Average MAE: {avg_mae:.2f}")
        print(f"Average MAPE: {avg_mape:.2f}%")

        results = open("ARIMA_results.csv", "a")
        results.write("{},{},{},{}\n".format(i, avg_mse, avg_mae, avg_mape))
        results.close()

def run_xgBoost():
    print("Reading Data...")
    data = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))
    data['Time'] = pd.to_datetime(data['Time'])

    # Add previous day's value
    data['Lag_1day'] = data['Time'].apply(lambda x: data.loc[data['Time'] == (x - pd.DateOffset(days=1)), 
                                                            'Load (kW)'].values[0] if (x - pd.DateOffset(days=1)) in data['Time'].values else None)
    data['Lag_2day'] = data['Time'].apply(lambda x: data.loc[data['Time'] == (x - pd.DateOffset(days=2)), 
                                                            'Load (kW)'].values[0] if (x - pd.DateOffset(days=2)) in data['Time'].values else None)
    data['Lag_1week'] = data['Time'].apply(lambda x: data.loc[data['Time'] == (x - pd.DateOffset(days=7)), 
                                                            'Load (kW)'].values[0] if (x - pd.DateOffset(days=7)) in data['Time'].values else None)
    # Remove first week
    cutoff = data['Time'].dt.date.min() + pd.DateOffset(days=7)
    data = data[data['Time'].dt.date >= cutoff.date()]

    # Clean time values
    data['Day'] = data['Time'].dt.day
    data['Month'] = data['Time'].dt.month
    data['Hour'] = data['Time'].dt.hour
    data['Year'] = data['Time'].dt.year
    data['Weekday'] = data['Time'].dt.weekday
    data.drop("Time", axis="columns", inplace=True)

    X = data.drop('Load (kW)', axis='columns', inplace=False)
    y = data['Load (kW)']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Set up the forecaster and fit the model
    print("Creating Model...")
    forecaster = XGBoostForecaster()
    print("Training Model...")
    forecaster.fit(X_train, y_train)

    # Make predictions
    print("Evaluating Model...")
    y_pred = forecaster.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")


if __name__ == "__main__":
    pass
