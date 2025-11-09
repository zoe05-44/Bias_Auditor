import numpy as np
import pandas as pd
import json

def compute_save_sample_weights(X_train, y_train, s_train, feature_names, save_path="outputs/results/sample_weights.json"):
    """
    Compute targeted probability sample weights and save to a JSON file.
    
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

    train_df['income'] = y_train
    train_df['sex'] = s_train


    # Assign weight to be 1 initially and boosted later 
    train_df['weight'] = 1.0

    #Boosted Group
    mask = (train_df['income'] ==1) & (train_df['sex'] == 0)
    ref_group = (train_df['income'] ==1) & (train_df['sex'] ==1)
    target_count= int(mask.sum())
    ref_count = int(ref_group.sum())
    n= len(train_df)
    target_prop = target_count/n

    print(f'Target Proportion: {target_prop}')
    boost_factor =  ref_count/target_count
    train_df.loc[mask, 'weight'] = boost_factor
    
    #Normalize weights to sum to n
    train_df['weight'] = train_df['weight']*n / train_df['weight'].sum()

    print(f'Total Sample: {n}')
    comp = train_df.groupby(['sex', 'income'])['weight'].sum()
    print(comp)
    target_effective_count = float(comp[(0,1)])
    sample_weights = train_df['weight'].tolist()



    #Convert weights to list and save as JSON
    reweighed = {'strategy': 'targeted_reweighting_high_income_women'}
    reweighed['sample_weights'] = sample_weights
    reweighed['boost_factor'] = boost_factor
    reweighed['target_group_count'] = target_count
    reweighed['target_group_effective_count'] =target_effective_count
    with open(save_path, 'w') as f:
         json.dump(reweighed, f)
    
    print(f"Sample weights saved to: {save_path}")
    
    return np.array(sample_weights)
