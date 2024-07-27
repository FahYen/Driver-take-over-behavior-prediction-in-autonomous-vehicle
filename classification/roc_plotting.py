import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# Load true labels
y_test = joblib.load('y_test.pkl')

# Binarize the output for multiclass ROC computation
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Adjust classes as per your target labels
n_classes = y_test_bin.shape[1]

# Initialize plot
plt.figure(figsize=(10, 8))

# Plot ROC curve for each model and each class
models = ["Logistic Regression", "Gradient Boosting", "Random Forest", "Naive Bayes", "AdaBoost", "RGF"]
for name in models:
    probs = joblib.load(f'{name}_probs.pkl')
    print(probs.shape)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.plot(fpr["micro"], tpr["micro"],
             label=f'{name} (micro-average ROC curve area = {roc_auc["micro"]:.2f})')

# Finalize plot
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-Average ROC Curves for Multiple Models')
plt.legend(loc="lower right")
plt.show()
