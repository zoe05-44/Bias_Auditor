import numpy as np
from sklearn.metrics import classification_report
import os

model = ['Gradient Boosting_reweighted_preds.npz','Logistic Regression_reweighted_preds.npz', 'Random Forest_reweighted_preds.npz']
for m in model: 
    path_npz = os.path.join('../outputs/preds/', m)
    logreg_data = np.load(path_npz)
    y_test_lr = logreg_data["y_test"]
    y_pred_lr = logreg_data["y_pred"]
    s_test_lr = logreg_data["s_test"]

    # Mask for high-income women
    high_income_women_mask = (s_test_lr == 0) & (y_test_lr== 1)


    # True labels and predicted labels
    y_true_hi_women = y_test_lr[high_income_women_mask]
    y_pred_hi_women_weigh = y_pred_lr[high_income_women_mask]

    # Print performance
    print(f"weighted model performance on high-income women {m}:")
    print(classification_report(y_true_hi_women, y_pred_hi_women_weigh, zero_division=0))

def evaluate_high_income_women(npz_path, label):
    data = np.load(npz_path)
    y_test = data["y_test"]
    y_pred = data["y_pred"]
    s_test = data["s_test"]

    mask = (s_test == 0) & (y_test == 1)
    y_true = y_test[mask]
    y_pred = y_pred[mask]

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "label": label,
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"]
    }

def explore_occupation_with_bias(df, occupation):
    occ_df = df[df['occupation'] == occupation]
    if occ_df.empty:
        print(f"No data found for occupation: {occupation}")
        return
    
    print(f"\n=== Occupation: {occupation} ===")
    
    print("\nSex distribution:")
    print(occ_df['sex'].value_counts(normalize=True))
    
    # Sex by labor effort group 
    print("\nSex by labor effort group:")
    print(occ_df.groupby('labor_effort_group')['sex'].value_counts(normalize=True))
    
    # Income distribution by effort & sex 
    print("\nIncome distribution by effort & sex:")
    print(occ_df.groupby(['labor_effort_group', 'sex'])['income'].value_counts(normalize=True))
    
    # Bias score calculation 
    p = occ_df['sex'].value_counts(normalize=True)
    female = occ_df[occ_df['sex'] == 'Female']
    male = occ_df[occ_df['sex'] == 'Male']
    
    p_f = p.get('Female', 0)
    r_f = len(female[female['income'] == '>50K']) / len(female) if len(female) > 0 else 0
    r_m = len(male[male['income'] == '>50K']) / len(male) if len(male) > 0 else 0
    
    bin_weights = {'low': 0.33, 'normal': 0.66, 'high': 1.0}
    def compute_effort_score(group):
        proportions = group['labor_effort_group'].value_counts(normalize=True)
        return sum(bin_weights.get(bin_label, 0) * proportions.get(bin_label, 0) for bin_label in bin_weights)
    
    e_f = compute_effort_score(female)
    e_m = compute_effort_score(male)
    
    numerator = r_f * e_f
    denominator = numerator + (r_m * e_m)
    score = numerator / denominator if denominator != 0 else None
    
    print("\n--- Bias Score ---")
    if score is None:
        print("Bias Score could not be computed")
    else:
        print(f"Bias Score: {score:.2f} | Female Representation: {p_f:.2f}")
        if score < p_f:
            print("→ Women are underrewarded relative to representation and effort")
        elif score > p_f:
            print("→ Women are rewarded more than representation and effort")
        else:
            print("→ Reward proportional to representation and effort")

