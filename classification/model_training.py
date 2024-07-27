import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from rgf.sklearn import RGFClassifier  # Ensure RGF is installed
from sklearn.preprocessing import StandardScaler

# Load the train and test sets
X_train = joblib.load('X_train.pkl')
X_test = joblib.load('X_test.pkl')
y_train = joblib.load('y_train.pkl')
y_test = joblib.load('y_test.pkl')

# Optional: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models with multiclass configuration
models = {
    "Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(n_estimators=100),
    "RGF": RGFClassifier()  # Make sure RGF is installed
}

# Train models and save probabilities
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train) # Standarize feature for logistic regression.
    else:
        model.fit(X_train, y_train) 

    probs = model.predict_proba(X_test_scaled if name == "Logistic Regression" else X_test)
    # print(name, probs.shape)
    joblib.dump(probs, f'{name}_probs.pkl')