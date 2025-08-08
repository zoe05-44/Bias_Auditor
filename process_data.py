import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def preprocessing_data(df):
    # Replace missing values and drop rows with NA
    df = df.replace('?', np.nan)
    df = df.dropna()
    df['income'] = df['income'].str.replace('.', '', regex=False)

    # map target and sensitive feature
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
    df['sex'] = df['sex'].map({'Female': 0, 'Male': 1})

    df = df.replace('?', np.nan)
    df = df.dropna()
    
    # Define features
    y = df['income']
    sensitive_feature = df['sex']
    X = df[['occupation', 'education', 'hours-per-week', 'age', 'workclass']]

    categorical_cols = ['occupation', 'education', 'workclass']
    numerical_cols = ['hours-per-week', 'age']

    # Define transformer
    transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, drop='first'), categorical_cols),
            ('num', SimpleImputer(strategy='mean'), numerical_cols)
        ],
        remainder='passthrough'  # no features dropped
    )

    X_transformed = transformer.fit_transform(X)

    #Spit data into training and test
    X_train, X_test,y_train, y_test, s_train, s_test = train_test_split(
        X_transformed, y, sensitive_feature, 
        test_size=0.2,
        random_state=42,
        stratify=y)

    return X_train, X_test,y_train, y_test, s_train, s_test

def get_feature_names(df):
    """
    Fit a column transformer on the relevant features to extract transformed feature names.
    Should match the logic used in preprocessing_data().
    
    Args:
        df (pd.DataFrame): The full (unprocessed) dataframe.

    Returns:
        feature_names (list): List of column names after transformation.
    """
    categorical_cols = ['occupation', 'education', 'workclass']
    numerical_cols = ['hours-per-week', 'age']

    # Create and fit transformer on raw columns
    dummy_encoder = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, drop='first'), categorical_cols),
            ('num', SimpleImputer(strategy='mean'), numerical_cols)
        ]
    )

    # Drop rows with NaNs to ensure fit works (same as in preprocessing)
    X_raw = df[['occupation', 'education', 'hours-per-week', 'age', 'workclass']].dropna()
    dummy_encoder.fit(X_raw)

    # Build full feature name list
    feature_names = (
        dummy_encoder.named_transformers_['cat']
        .get_feature_names_out(categorical_cols)
        .tolist()
        + numerical_cols
    )

    return feature_names



