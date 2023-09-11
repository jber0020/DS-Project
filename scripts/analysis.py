"""
ANALYSIS FILE

FIT3164
Project 4 - Electricity Demand Forecasting
Team 4
- Joshua Berg
- Ryan Hendler
- Yuechuan Li
- Yangyi Li

This file contains code that analysises the data, and the models' performance.
"""

# Imports
import pandas as pd
from utils import *
import matplotlib.pyplot as plt
import numpy
from wrangling import *
from sklearn.feature_selection import mutual_info_regression
from models import *
import math


def summary(data):
    for col in data.columns:
        print("=========================")
        print("Variable:", col)
        print("Min:", min(data[col]))
        print("Median:", numpy.median(data[col]))
        print("Mean:", numpy.mean(data[col]))
        print("Max:", max(data[col]))
        print("Std:", numpy.std(data[col]))
        print("Empty Values:", data[col].isna().sum())
        plt.boxplot(data[col])
        # plt.show()
        plt.title(col)
        plt.savefig(get_root('modelling/plots/{}.png'.format(col)))
        plt.close()

def mutual_info_reg(data):
    x = data.drop('Load (kW)', axis=1)
    y = data['Load (kW)']

    x = (x-x.mean())/x.std()
    y = (y-y.mean())/y.std()

    mir_scores = mutual_info_regression(x, y)
    mir_scores = pd.Series(mir_scores)
    mir_scores.index = x.columns
    print("Mutual Information Regression")
    print(mir_scores.sort_values(ascending=False))

def model_performance():
    # Read Data
    data = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))
    
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
    
    results = open(get_root('scripts/results/comparison_results.txt'), 'w')
    results.write("model,mse,mae,mape\n")
    # Run Benchmarks
    # Naive
    pred = y.shift(1)
    naive_res = naive(y, 1)
    mse, mae, mape = naive_res[0], naive_res[1], naive_res[2]
    results.write("1HourNaive,{},{},{}\n".format(mse, mae, mape))

    # Seasonal Naive - Day
    naive_res = naive(y, 24)
    mse, mae, mape = naive_res[0], naive_res[1], naive_res[2]
    results.write("24HourNaive,{},{},{}\n".format(mse, mae, mape))


    # Seasonal Naive - 2 Day
    naive_res = naive(y, 48)
    mse, mae, mape = naive_res[0], naive_res[1], naive_res[2]
    results.write("48HourNaive,{},{},{}\n".format(mse, mae, mape))


    # Seasonal Naive - Week
    naive_res = naive(y, 25*7)
    mse, mae, mape = naive_res[0], naive_res[1], naive_res[2]
    results.write("1WeekNaive,{},{},{}\n".format(mse, mae, mape))


    # Seasonal Naive - Month
    naive_res = naive(y, 1)
    mse, mae, mape = naive_res[0], naive_res[1], naive_res[2]
    results.write("1HourNaive,{},{},{}\n".format(mse, mae, mape))


    # Seasonal Naive - Year
    naive_res = naive(y, 24*365)
    mse, mae, mape = naive_res[0], naive_res[1], naive_res[2]
    results.write("1YearNaive,{},{},{}\n".format(mse, mae, mape))

    # # Random Walk with Drift
    # np.random.seed(52)
    # n_steps = 48
    # drift = 1
    # sigma = y.std()
    # mapes = []
    # maes = []
    # mses = []

    # # Define Random Walk with Drift
    # def random_walk(drift):
    #     for i in range(len(y_test)-48):
    #         walk = [y[i]]
    #         for _ in range(n_steps):
    #             walk.append(walk[-1] + drift + np.random.normal(scale=sigma))
    #         actuals = y[i+1:i+49]
    #         mses.append(mean_squared_error(actuals, walk[1:]))
    #         maes.append(mean_absolute_error(actuals, walk[1:]))
    #         mapes.append(mean_absolute_percentage_error(actuals, walk[1:])*100)
    #     return np.mean(mses)
    
    # # Optimise Drift (Uncomment to Run, takes ages to optimise)
    # # drift = minimize(random_walk, drift).x[0]
    # # print("Best Drift:", drift)

    # mapes = []
    # maes = []
    # mses = []

    # # Run Optimised Random Walk
    # random_walk(drift)
    # print("===Random Walk with Drift===")
    # print("MSE:", np.mean(mses))
    # print("MAE:", np.mean(maes))
    # print("MAPE:", np.mean(mapes))
    # results.write("Random Walk\n")
    # results.write('MSE:{}, MAE: {}, MAPE: {}\n'.format(np.mean(mses), np.mean(maes), np.mean(mapes)))

    # Run ANN Model 1
    NetModel = NeuralNetForecaster()
    NetModel.build_model(input_size=len(X1_train.columns),
                         layers=[64,32,16],
                         learning_rate=0.1)
    NetModel.train_model(X1_train, y_train, X1_val, y_val)
    net_results = NetModel.test_model(X1_test, y_test)
    results.write("NN1,{},{},{}\n".format(net_results[1], net_results[2], net_results[3]))

    # Run ANN Model 2
    NetModel = NeuralNetForecaster()
    NetModel.build_model(input_size=len(X2_train.columns),
                         layers=[64,32,16],
                         learning_rate=0.1)
    NetModel.train_model(X2_train, y_train, X2_val, y_val)
    net_results = NetModel.test_model(X2_test, y_test)
    results.write("NN2,{},{},{}\n".format(net_results[1], net_results[2], net_results[3]))

    # Run XGBoost Model 1
    XGBoost = xgBoostForecaster()
    XGBoost.build_model()
    XGBoost.train_model(X1_train, y_train)
    xgb_results = XGBoost.test_model(X1_test, y_test)
    results.write("XGB1,{},{},{}\n".format(
        xgb_results[0],
        xgb_results[1],
        xgb_results[2]
    ))
    XGBoost.save_model('xgb1', 'scripts/models')

    # Run XGBoost Model 2
    XGBoost = xgBoostForecaster()
    XGBoost.build_model()
    XGBoost.train_model(X2_train, y_train)
    xgb_results = XGBoost.test_model(X2_test, y_test)
    results.write("XGB2,{},{},{}\n".format(
        xgb_results[0],
        xgb_results[1],
        xgb_results[2]
    ))
    XGBoost.save_model('xgb2', 'scripts/models')

    # Run Random Forest Model 1
    Rf = RandomForestForecaster()
    Rf.build_model()
    Rf.train_model(X1_train, y_train)
    Rf_results = Rf.test_model(X1_test, y_test)
    results.write("RF1,{},{},{}\n".format(
        Rf_results[0],
        Rf_results[1],
        Rf_results[2]
    ))

    # Run Random Forest Model 2
    Rf = RandomForestForecaster()
    Rf.build_model()
    Rf.train_model(X2_train, y_train)
    Rf_results = Rf.test_model(X2_test, y_test)
    results.write("RF2,{},{},{}\n".format(
        Rf_results[0],
        Rf_results[1],
        Rf_results[2]
    ))

    results.close()

if __name__=="__main__":
    model_performance()
