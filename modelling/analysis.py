"""
"""

# Imports
import pandas as pd
from utils import *
import matplotlib.pyplot as plt
import numpy
from wrangling import *
from sklearn.feature_selection import mutual_info_regression


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

if __name__=="__main__":
    # Read Data
    data = pd.read_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'))
    summary(data)
    mutual_info_reg(data)