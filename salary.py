import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder

# --------------------------------------
# Load and split data
# --------------------------------------
_df_jobs_clean = pd.read_csv('data/jobs_clean_nt.csv')
df_X = _df_jobs_clean.drop(columns=["normalized_salary_log10", "normalized_salary", "min_salary", "max_salary", "med_salary"])
df_y = _df_jobs_clean["normalized_salary_log10"]

df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)


# --------------------------------------
# Functions for train and test datasets
# --------------------------------------
def get_train_dataset():
    return df_X_train, df_y_train

def get_test_dataset():
    return df_X_test, df_y_test

def evaluate_train_test_predictions(y_train_pred, y_test_pred):
    y_train = df_y_train.values
    y_test = df_y_test.values
    y_train_exp = 10**y_train
    y_test_exp = 10**y_test
    y_train_pred_exp = 10**np.array(y_train_pred)
    y_test_pred_exp = 10**np.array(y_test_pred)
    y_train_exp_mean = np.mean(y_train_exp)

    train_r2 = r2_score(y_train_exp, y_train_pred_exp)
    test_r2 = r2_score(y_test_exp, y_test_pred_exp)
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")

    train_rmse = np.sqrt(mean_squared_error(y_train_exp, y_train_pred_exp))
    test_rmse = np.sqrt(mean_squared_error(y_test_exp, y_test_pred_exp))
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    train_mae = mean_absolute_error(y_train_exp, y_train_pred_exp)
    test_mae = mean_absolute_error(y_test_exp, y_test_pred_exp)
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    test_mae_mean = mean_absolute_error(y_test_exp, [y_train_exp_mean]*len(y_test_exp))
    improvement = ((test_mae_mean-test_mae) / test_mae_mean)*100
    print(f'On average, our predicted salaries are ${test_mae:.2f} off the true salaries')
    print(f'This is {improvement:.2f}% better than a naive global mean')



## --------------------------------------
## Functions to encode features
## --------------------------------------

experience_level_encoder = OrdinalEncoder(
    categories=[[
        "Unknown",
        "Internship",
        "Entry level",
        "Associate",
        "Mid-Senior level",
        "Director",
        "Executive"
    ]]
)

work_type_encoder = OrdinalEncoder(
    categories=[[
        "Other",
        "Volunteer",
        "Internship",
        "Temporary",
        "Part-time",
        "Contract",
        "Full-time"
    ]]
)
