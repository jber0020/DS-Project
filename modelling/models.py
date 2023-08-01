"""

"""

# Imports
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import *
from wrangling import *

class ANN:
    """
    
    """

    def __init__(self, input_size, layers, batch_size=32, learning_rate=0.001, epochs=100, metrics=['mae']) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(layers[0], input_dim=input_size, activation='relu'))
        if len(layers) > 1:
            for i in layers[1:]:
                self.model.add(tf.keras.layers.Dense(i, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=metrics)

    def train_model(self, X_train, y_train):
        history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)
        return self.model, history

    def test_model(self, X_val, y_val):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def predict(self):
        pass

if __name__ == "__main__":
    data = pd.read_excel(get_root('data/elec_p4_dataset/Train/Actuals_part1.xlsx'))
    data = clean_data(data)

    X, y = data.drop("Load (kW)", axis="columns"), data["Load (kW)"]

    Model = ANN(input_size = len(X.columns), layers = [64, 32, 16], metrics=['mae', 'mape'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    model, history = Model.train_model(X_train, y_train)

    train_mae = history.history['mae']
    train_mape = history.history['mape']

    mae_df = pd.DataFrame({'train_mae': train_mae, 'train_mape': train_mape})
    mae_df.to_csv(get_root("history.csv"))
