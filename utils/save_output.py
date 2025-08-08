import numpy as np
import os 
import json

def save_metrics(metrics_dict, model_name, path=None):
    if path is None:
        path = os.path.join("outputs", "results", "fairness_metrics.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if hasattr(metrics_dict, "to_dict"):
        metrics_dict = metrics_dict.to_dict()
    
        if os.path.exists(path):
            with open(path, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
    else:
        existing_data = {}

    existing_data[model_name] = metrics_dict

    with open(path, 'w') as f:
        json.dump(existing_data, f, indent=3)
        
  

def save_predictions_npz(y_test, y_pred, sensitive_features, path):
    print('searching for file')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, y_test=y_test, y_pred=y_pred, s_test=sensitive_features)
    print('Data added to file')

def save_model_result(model_name, report, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {model_name: report}
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)