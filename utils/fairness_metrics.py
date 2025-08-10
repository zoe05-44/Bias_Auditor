from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from utils import save_output as s
from sklearn.metrics import confusion_matrix
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score


def compute_fairness(y_true, y_pred, sensitive_features, output_path=None):
    metric_frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    df = metric_frame.by_group    
    return df

def evaluate_model(model,model_name, X_train, y_train, X_test, y_test, s_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #save predictions
    s.save_predictions_npz(y_test, y_pred, s_test, path=f"outputs/preds/{model_name}_preds.npz")

    # Save fairness metrics for every model
    fairness_df = compute_fairness(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=s_test
    )
    s.save_metrics(fairness_df, model_name)

    #return classification report for saving
    return classification_report(y_test, y_pred, output_dict=True)

def tune_gradient_boosting(model_name,X_train, y_train, X_test, y_test, s_test, cv, pipe_gb):

    param_grid = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.05, 0.1, 0.2],
        'model__max_depth': [3, 5]
    }

    grid = GridSearchCV(pipe_gb, param_grid, scoring='f1', cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best params for Gradient Boosting:")
    print(grid.best_params_)

    y_pred = grid.best_estimator_.predict(X_test)
    print("Test set evaluation after tuning:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save fairness metrics for tuned model
    fairness_df = compute_fairness(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=s_test,
        output_path="outputs/fairness_report.csv"
    )
    s.save_metrics(fairness_df, model_name)
    return fairness_df

