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
import numpy as np
from scipy.optimize import minimize


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

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
    # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}
            self.column_indices = {name: i for i, name in
                                enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
  
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

class LSTM:
    """
    LSTM RNN Model
    """
    def __init__(self, input_shape, num_features, layers = [128], epochs=100, batch_size=32, metrics=['mae', 'mse', 'mape']):
        self.batch_size = batch_size
        self.epochs = epochs
        self.metrics= metrics
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(units = layers[0], input_shape=input_shape, return_sequences=False),
            tf.keras.layers.Dense(units=48, kernel_initializer=tf.initializers.zeros())
        ])
        self.model.compile(loss="mean_squared_error", metrics=self.metrics, optimizer=tf.keras.optimizers.Adam())

    def LSTM_clean_data(data):
        data['Time'] = pd.to_datetime(data['Time'])

        data['Lag_1week'] = data['Time'].apply(lambda x: data.loc[data['Time'] == (x - pd.DateOffset(days=7)), 
                                                            'Load (kW)'].values[0] if (x - pd.DateOffset(days=7)) in data['Time'].values else None)

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
    
    def train_model(self, train, val):
        self.model.fit(train, epochs=self.epochs, validation_data=val, 
                       callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode='min')])
    
    def test_model(self, X_val, y_val):
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
    data = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))

    X, y = data.drop("Load (kW)", axis="columns"), data["Load (kW)"]

    model = ARIMA(data=y, p=24, d=1, q=0)

def LSTM_Tuning():
    data = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))

    data = LSTM.LSTM_clean_data(data)

    n = len(data)

    train_df = data[0:int(n*0.7)]
    val_df = data[int(n*0.7):int(n*0.9)]
    test_df = data[int(n*0.9):]

    num_features = data.shape[1]

    multi_window = WindowGenerator(input_width=24,
                                   label_width=48,
                                   shift=48,
                                   train_df=train_df,
                                   val_df=val_df,
                                   test_df=test_df,
                                   label_columns=['Load (kW)'])

    Model = LSTM(input_shape=(24, num_features), num_features=num_features)

    Model.train_model(multi_window.train, multi_window.val)


if __name__ == "__main__":
    LSTM_Tuning()
