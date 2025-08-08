import numpy as np
import pandas as pd
import json

def compute_save_sample_weights(X_train, y_train, s_train, feature_names, save_path="outputs/results/sample_weights.json"):
    """
    Compute inverse probability sample weights and save to a JSON file.
    
    Args:
        X_train (array-like): Feature matrix.
        y_train (pd.Series or array): Binary target values (0 or 1).
        s_train (pd.Series or array): Binary sensitive attribute (e.g. gender).
        feature_names (list): Names of the columns in X_train.
        save_path (str): File path to save the weights (.json).
    
    Returns:
        sample_weights (np.ndarray): Array of normalized weights.
    """
    
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df = X_train_df.copy()
    train_df['income'] = y_train.values
    train_df['sex'] = s_train.values
    
    # Compute (group, label) joint probability
    group_counts = train_df.groupby(['sex', 'income']).size()
    joint_prob = group_counts / len(train_df)
    
    # Step 3: Assign inverse probability weights
    train_df['weight'] = train_df.apply(
        lambda row: 1 / joint_prob[(row['sex'], row['income'])],
        axis=1
    )
    
    #Normalize weights to sum to 1
    train_df['weight'] /= train_df['weight'].sum()
    
    #Convert weights to list and save as JSON
    sample_weights = train_df['weight'].tolist()
    with open(save_path, 'w') as f:
        json.dump({'sample_weights': sample_weights}, f)
    
    print(f"Sample weights saved to: {save_path}")
    return np.array(sample_weights)
