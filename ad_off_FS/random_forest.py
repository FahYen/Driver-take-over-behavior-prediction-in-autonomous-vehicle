# https://github.com/scikit-learn-contrib/boruta_py
# Needs to tune max_depth with cross validation
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

import pandas as pd
###
### Local Imports
###
from preprocess import AD_off, meanRR, maxHR, meanHR, RMSSD, SDHR, SDNN, HF, LF, total_power

# load x and y
X = pd.concat([meanRR, maxHR, meanHR, RMSSD, SDHR, SDNN, HF, LF, total_power], axis=1).values
y = AD_off.values

# create regression model
forest = RandomForestRegressor(n_jobs=-1, max_depth=20) 
# max_depth set to 5 to prevent overfitting
# 5 is just a tentative number, further tuning is needed with more features

feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2)

# find all relevant features
feat_selector.fit(X, y)

# check selected features
feat_selector.support_

# check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)

print("Selected Features: ", feat_selector.support_)
print("Feature Ranking: ", feat_selector.ranking_)

# ranking
# max_depth=5
# [3 2 3]
# max_depth=10
# [4 2 3]
# max_depth=20
# [3 2 4]