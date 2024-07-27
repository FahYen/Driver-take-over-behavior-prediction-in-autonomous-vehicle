import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Read features
df_headers = pd.read_csv("/Users/lambert/NHanCE/Complete_Data/all_feature_nosubjective.csv", nrows=0)
cols_to_exclude = ['id', 'task', 'mean_scl', 'stdev_scl', 'RT_adoff', 'RT_steer', 'mean_hand', 'mean_eye', 'big_maneuver']
x_use_columns = [col for col in df_headers.columns if col not in cols_to_exclude]
X = pd.read_csv("/Users/lambert/NHanCE/Complete_Data/all_feature_nosubjective.csv", usecols=x_use_columns)
X = X.to_numpy()

# Read y. Change variable inside [ ] to change y
y = pd.read_csv("/Users/lambert/NHanCE/Complete_Data/all_feature_nosubjective.csv",usecols=['RT_steer'])
y = y.values.ravel()
y_binned = pd.qcut(y, q=3, labels=False)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.3, random_state=8)

# download train and test split data into disk
joblib.dump(X_train, 'X_train.pkl')
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(y_test, 'y_test.pkl')