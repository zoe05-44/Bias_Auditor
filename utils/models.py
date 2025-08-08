from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

models_baseline = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

pipe_logreg = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', LogisticRegression(max_iter=2000))
])

# SMOTE only for tree models
pipe_rf = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier())
])

pipe_gb = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', GradientBoostingClassifier())
])

models_smote = {
    'Logistic Regression': pipe_logreg,
    'Random Forest': pipe_rf,
    'Gradient Boosting': pipe_gb
}