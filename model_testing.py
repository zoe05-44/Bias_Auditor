from process_data import preprocessing_data
from data import df
from sklearn.model_selection import StratifiedKFold, cross_val_score
from utils import save_output as s
from utils import models as m
from utils import fairness_metrics as f
import numpy as np

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


#Main execution
X_train, X_test, y_train, y_test, s_train, s_test = preprocessing_data(df)

models = m.models_smote

for name, model in models.items():
    # Cross-validated F1 score on training set
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    print(f"\n{name} | CV F1 scores: {scores}")
    print(f"{name} | Mean CV F1: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    # Evaluation on test set
    print(f"Final Evaluation on Test Set for {name}")
    result = f.evaluate_model(model, name, X_train, y_train, X_test, y_test, s_test)
    #Save classification result
    s.save_model_result(name, result, path = f"outputs/results/{name}.json")

# Fine-tune and evaluate the best model
f.tune_gradient_boosting(name, X_train, y_train, X_test, y_test, s_test, cv, m.pipe_gb)