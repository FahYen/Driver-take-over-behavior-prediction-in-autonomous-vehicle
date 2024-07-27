import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from preprocess_complete import X, maneuver

y = maneuver

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create logistic regression model
model = LogisticRegression()

# Recursive Feature Elimination
rfe = RFE(model, n_features_to_select=1)
rfe = rfe.fit(X_train, y_train)

# Print rankings
print("Features sorted by their rank:")
feature_rankings = sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), X.columns))
for rank, feature in feature_rankings:
    print(f"{feature}: {rank}")