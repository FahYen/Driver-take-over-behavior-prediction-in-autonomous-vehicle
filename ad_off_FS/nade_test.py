import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression

# Re-import the dataset since the environment was reset
file_path = 'year3_data\RT_adoff.csv'
data = pd.read_csv(file_path)

# Prepare the dataset excluding the 'task' column
X_no_task = data.drop(['RT_adoff', 'task'], axis=1)
y = data['RT_adoff']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_no_task)

# Random Forest for feature importances
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_scaled, y)
rf_feature_importances = rf_model.feature_importances_

# Initialize the Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=0)
gbr_model.fit(X_scaled, y)
gbr_feature_importances = gbr_model.feature_importances_

# Initialize LassoCV for feature selection
lasso_model = LassoCV(cv=5, random_state=0)
lasso_model.fit(X_scaled, y)
# Get the coefficients from the Lasso model
lasso_coefs = np.abs(lasso_model.coef_)

# Calculate mutual information
mi = mutual_info_regression(X_scaled, y)
mi /= np.max(mi)  # Normalize the mutual information scores

# SVM
svr_model = SVR(kernel='linear')
svr_model.fit(X_scaled, y)
svr_coefs = np.abs(svr_model.coef_[0])

# Logistic Regression (use median as split point)
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_scaled, (y > y.median()).astype(int))  # Converting to binary for logistic regression
log_reg_coefs = np.abs(log_reg_model.coef_[0])

# Compile all the feature importances into a single DataFrame
feature_selection_df = pd.DataFrame({
    'Feature': X_no_task.columns,
    'RandomForestImportance': rf_feature_importances,
    'GradientBoostingImportance': gbr_feature_importances,
    'LassoCoefficients': lasso_coefs,
    'MutualInformation': mi,
    'SVR_Coefficients': svr_coefs,
    'LogReg_Coefficients': log_reg_coefs
})

# We take the absolute value of coefficients as importance score for Lasso
feature_selection_df['LassoCoefficients'] = feature_selection_df['LassoCoefficients'].abs()

# Sort the DataFrame based on the Random Forest feature importance
feature_selection_df = feature_selection_df.sort_values(by='RandomForestImportance', ascending=False)

# Select the top 10 features from each method
top_features_rf = feature_selection_df.nlargest(10, 'RandomForestImportance')
top_features_gbr = feature_selection_df.nlargest(10, 'GradientBoostingImportance')
top_features_lasso = feature_selection_df.nlargest(10, 'LassoCoefficients')
top_features_mi = feature_selection_df.nlargest(10, 'MutualInformation')
top_features_svr = feature_selection_df.nlargest(10, 'SVR_Coefficients')
top_features_log_reg = feature_selection_df.nlargest(10, 'LogReg_Coefficients')

# Combine the top 10 features into a single DataFrame for comparison
top_features_comparison = pd.DataFrame({
    'Feature_RF': top_features_rf['Feature'].values,
    'Importance_RF': top_features_rf['RandomForestImportance'].values,
    'Feature_GBR': top_features_gbr['Feature'].values,
    'Importance_GBR': top_features_gbr['GradientBoostingImportance'].values,
    'Feature_Lasso': top_features_lasso['Feature'].values,
    'Importance_Lasso': top_features_lasso['LassoCoefficients'].values,
    'Feature_MI': top_features_mi['Feature'].values,
    'Importance_MI': top_features_mi['MutualInformation'].values,
    'Feature_SVR': top_features_svr['Feature'].values,
    'Importance_SVR': top_features_svr['SVR_Coefficients'].values,
    'Feature_LogReg': top_features_log_reg['Feature'].values,
    'Importance_LogReg': top_features_log_reg['LogReg_Coefficients'].values
})


print(top_features_comparison)
