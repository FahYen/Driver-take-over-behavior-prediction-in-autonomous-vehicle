from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
###
### Local Imports
###
from preprocess_complete import X, y

# create regression model
forest = RandomForestRegressor(n_jobs=-1, max_depth=5) 
feat_selector = BorutaPy(verbose=2, estimator=forest, n_estimators='auto')

# find all relevant features
feat_selector.fit(X.to_numpy(), y)

# Store the results in a list
feature_rankings = []
for i in range(len(feat_selector.support_)):
    feature_name = X.columns[i]
    passes_test = "Passes the test" if feat_selector.support_[i] else "Doesn't pass the test"
    ranking = feat_selector.ranking_[i]
    feature_rankings.append((feature_name, passes_test, ranking))

# Sort the list by ranking
feature_rankings.sort(key=lambda x: x[2])

# Print the sorted results
print("\n------Feature Ranking------\n")
for feature in feature_rankings:
    print(f"{feature[1]}: {feature[0]} : {feature[2]}")