import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from bayes_opt import BayesianOptimization
"""pip install bayesian-optimization"""

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

df = pd.read_csv("flight_delays_train.csv")
print(df.shape)
print(df.info())

df["dep_delayed_15min"] = df["dep_delayed_15min"].map(df["dep_delayed_15min"].value_counts(normalize=True).to_dict())
print(df["dep_delayed_15min"])

train_df["hour"] = train_df["DepTime"] // 100
print(train_df["hour"])

train_df['hour'] = train_df['DepTime'] // 100
train_df.loc[train_df['hour'] == 24, 'hour'] = 0
train_df.loc[train_df['hour'] == 25, 'hour'] = 1
train_df['minute'] = train_df['DepTime'] % 100

test_df['hour'] = test_df['DepTime'] // 100
test_df.loc[test_df['hour'] == 24, 'hour'] = 0
test_df.loc[test_df['hour'] == 25, 'hour'] = 1
test_df['minute'] = test_df['DepTime'] % 100

# Season
train_df['summer'] = (train_df['Month'].isin([6, 7, 8])).astype(np.int32)
train_df['autumn'] = (train_df['Month'].isin([9, 10, 11])).astype(np.int32)
train_df['winter'] = (train_df['Month'].isin([12, 1, 2])).astype(np.int32)
train_df['spring'] = (train_df['Month'].isin([3, 4, 5])).astype(np.int32)


test_df['summer'] = (test_df['Month'].isin([6, 7, 8])).astype(np.int32)
test_df['autumn'] = (test_df['Month'].isin([9, 10, 11])).astype(np.int32)
test_df['winter'] = (test_df['Month'].isin([12, 1, 2])).astype(np.int32)
test_df['spring'] = (test_df['Month'].isin([3, 4, 5])).astype(np.int32)

# Daytime
train_df['daytime'] = pd.cut(train_df['hour'], bins=[0, 6, 12, 18, 23], include_lowest=True)
test_df['daytime'] = pd.cut(test_df['hour'], bins=[0, 6, 12, 18, 23], include_lowest=True)

# Extract the labels
train_y = train_df.pop('dep_delayed_15min')
train_y = train_y.map({'N': 0, 'Y': 1})

# Concatenate for preprocessing
train_split = train_df.shape[0]
full_df = pd.concat((train_df, test_df))
full_df['Distance'] = np.log(full_df['Distance'])