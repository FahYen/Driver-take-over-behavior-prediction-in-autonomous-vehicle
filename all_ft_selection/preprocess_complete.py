import pandas as pd

# Read features
df_headers = pd.read_csv("/Users/lambert/NHanCE/Complete_Data/all_feature_nosubjective.csv", nrows=0)
cols_to_exclude = ['id', 'task', 'mean_scl', 'stdev_scl', 'RT_adoff', 'RT_steer', 'mean_hand', 'mean_eye', 'big_maneuver']
x_use_columns = [col for col in df_headers.columns if col not in cols_to_exclude]
X = pd.read_csv("/Users/lambert/NHanCE/Complete_Data/all_feature_nosubjective.csv", usecols=x_use_columns)

# Read y. Change variable inside [ ] to change y
y = pd.read_csv("/Users/lambert/NHanCE/Complete_Data/all_feature_nosubjective.csv",usecols=['RT_steer'])
y = y.values.ravel()
maneuver = pd.read_csv("/Users/lambert/NHanCE/Complete_Data/all_feature_nosubjective.csv",usecols=['big_maneuver'])


