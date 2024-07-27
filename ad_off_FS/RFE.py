import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR  # Support Vector Regression for regression tasks
from sklearn.model_selection import cross_val_score

### local imports
from preprocess import AD_off, meanRR, maxHR, meanHR, RMSSD, SDHR, SDNN, HF, LF, total_power

# Custom wrapper function to evaluate feature importance for non-linear SVM in regression
def custom_feature_importance(X, y, estimator):
    scores = []
    for i in range(X.shape[1]):
        X_subset = X[:, i].reshape(-1, 1)
        score = np.mean(cross_val_score(estimator, X_subset, y, scoring='neg_mean_squared_error', cv=2))
        scores.append(score)
    return np.array(scores)

def perform_rfe(X, y):
    # Initialize SVR with a non-linear kernel
    svr = SVR(kernel="rbf")
    
    # Flatten y into a 1D array
    y = y.ravel()
    
    # Get initial feature importances
    importances = custom_feature_importance(X, y, svr)
    
    # Sort features by importance
    sorted_idx = np.argsort(importances)
    
    print("Sorted feature indices:", sorted_idx)

def main():
    X = np.column_stack([meanRR, maxHR, meanHR, RMSSD, SDHR, SDNN, HF, LF, total_power])
    y = AD_off.values
    
    perform_rfe(X, y)

if __name__ == '__main__':
    main()
