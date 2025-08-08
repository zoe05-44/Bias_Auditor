import numpy as np
from sklearn.metrics import classification_report
import os

model = ['Gradient Boosting_reweighted_preds.npz','Logistic Regression_reweighted_preds.npz', 'Random Forest_reweighted_preds.npz']
for m in model: 
    path_npz = os.path.join('../outputs/preds/', m)
    logreg_data = np.load(path_npz)
    y_test_lr = logreg_data["y_test"]
    y_pred_lr = logreg_data["y_pred"]
    s_test_lr = logreg_data["s_test"]

    # Mask for high-income women
    high_income_women_mask = (s_test_lr == 0) & (y_test_lr== 1)


    # True labels and predicted labels
    y_true_hi_women = y_test_lr[high_income_women_mask]
    y_pred_hi_women_weigh = y_pred_lr[high_income_women_mask]

    # Print performance
    print(f"weighted model performance on high-income women {m}:")
    print(classification_report(y_true_hi_women, y_pred_hi_women_weigh, zero_division=0))

def evaluate_high_income_women(npz_path, label):
    data = np.load(npz_path)
    y_test = data["y_test"]
    y_pred = data["y_pred"]
    s_test = data["s_test"]

    mask = (s_test == 0) & (y_test == 1)
    y_true = y_test[mask]
    y_pred = y_pred[mask]

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "label": label,
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"]
    }
