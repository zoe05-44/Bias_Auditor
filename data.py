from ucimlrepo import fetch_ucirepo 
import pandas as pd
import os
  
def fetch_dataset():
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week',
        'native-country', 'income'
    ]

    # Define the path to CSV file.
    file_path = 'adult.csv'

    try:
        df = pd.read_csv(
            file_path,
            header=None,           
            names=column_names,    
            sep=r',\s*',           
            engine='python',       
            na_values='?'          
        )
        print("Successfully loaded data from 'data/adult.csv'.")
        print("DataFrame shape:", df.shape)
        return df
    

    except FileNotFoundError:
        print(f"Error: The file was not found at '{file_path}'.")
        print("Please make sure you have created the file 'adult.csv'")


df = fetch_dataset()