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

    # 2. Define the path to your local CSV file.
    file_path = 'adult.csv'

    # 3. Read the CSV file into a pandas DataFrame.
    try:
        df = pd.read_csv(
            file_path,
            header=None,           # Tell pandas there is no header row in the file.
            names=column_names,    # Provide our list of column names.
            sep=r',\s*',           # This handles the inconsistent spacing after the commas.
            engine='python',       # The 'python' engine is needed for the separator above.
            na_values='?'          # Automatically recognize '?' as a missing value.
        )
        print("Successfully loaded data from 'data/adult.csv'.")
        print("DataFrame shape:", df.shape)
        return df
    

    except FileNotFoundError:
        print(f"Error: The file was not found at '{file_path}'.")
        print("Please make sure you have created the file 'adult.csv'")
    """adult = fetch_ucirepo(id=2) 
    return adult.data.original
df = fetch_dataset()"""

df = fetch_dataset()