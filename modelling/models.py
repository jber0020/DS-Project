"""

"""

# Imports
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import *
from wrangling import *
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA as sARIMA

class xgBoost:
    """
    xgBoost Model
    """
    def __init__(self) -> None:
        pass


class ARIMA:
    """
    ARIMA Model
    """
    def __init__(self, data, p, d, q) -> None:
        self.model = sARIMA(data, order=(p, d, q))
        self.fit = self.model.fit()
        print(self.fit.summary())
        pass

class LSTM:
    """
    LSTM RNN Model
    """
    def __init__(self):
        pass

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
    print("===Naive Model===")
    print("MSE:", mean_squared_error(y[1:], pred[1:]))
    print("MAE:", mean_absolute_error(y[1:], pred[1:]))
    print("MAPE:", mean_absolute_percentage_error(y[1:], pred[1:])*100)

    # Seasonal Naive - Day
    pred = y.shift(24)
    print("===24h Naive Model===")
    print("MSE:", mean_squared_error(y[24:], pred[24:]))
    print("MAE:", mean_absolute_error(y[24:], pred[24:]))
    print("MAPE:", mean_absolute_percentage_error(y[24:], pred[24:])*100)

    # Seasonal Naive - Week
    pred = y.shift(24*7)
    print("===1week Naive Model===")
    print("MSE:", mean_squared_error(y[24*7:], pred[24*7:]))
    print("MAE:", mean_absolute_error(y[24*7:], pred[24*7:]))
    print("MAPE:", mean_absolute_percentage_error(y[24*7:], pred[24*7:])*100)

def run_ANN():
    data = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))

    X, y = data.drop("Load (kW)", axis="columns"), data["Load (kW)"]

    X = (X-X.mean()) / X.std()

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
            results.write("{},{},{},{},{},{}\n".format(lr, l, val_loss, mse, mae, mape))
            results.close()

def run_ARIMA():
    data = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))

    X, y = data.drop("Load (kW)", axis="columns"), data["Load (kW)"]

    model = ARIMA(data=y, p=24, d=1, q=0)



if __name__ == "__main__":
    run_ARIMA()
