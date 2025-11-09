from process_data import preprocessing_data, get_feature_names
from data import df
from utils import save_output as s
from sklearn.model_selection import StratifiedKFold
import os
from utils import models as m
from utils import fairness_metrics as f
from reweighting import compute_save_sample_weights
import numpy as np
import json
weights_path = "outputs/results/sample_weights.json"
splits_path = "outputs/preds/train_test_split.npz"

# Load Preprocessed Data 
if not os.path.exists(splits_path):
    print(f"ERROR: Splits file not found at {splits_path}")
    exit(1)
print("Loading train/test splits...")
splits = np.load(splits_path)
X_train = splits['X_train']
X_test = splits['X_test']
y_train = splits['y_train']
y_test = splits['y_test']
s_train = splits['s_train']
s_test = splits['s_test']

print(f"✓ Loaded splits - Train size: {len(X_train)}, Test size: {len(X_test)}")

feature_names = get_feature_names(df)

# Load Precomputed Sample Weights
if os.path.exists(weights_path) and os.stat(weights_path).st_size > 0:
    with open(weights_path, "r") as o:
        data = json.load(o)    
else:
    print("Weights file empty...generating weights...")
    compute_save_sample_weights(X_train, y_train, s_train, feature_names)

    with open(weights_path, "r") as o:
        data = json.load(o)


sample_weights = np.array(data["sample_weights"])

# Cross Validation Setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


models = m.models_baseline 

#Train and evaluate each model with sample weights 
for name, model in models.items():
    print(f"\n{name} | Training with reweighting...")

    model.fit(X_train, y_train, sample_weight=sample_weights)


    # Evaluate on test set
    result = f.evaluate_model(
        model,
        name + "_reweighted",
        X_train, y_train,
        X_test, y_test,
        s_test
    )

    #Save result
    s.save_model_result(
        name + "_reweighted",
        result,
        path=f"outputs/results/{name}_reweighted.json")
