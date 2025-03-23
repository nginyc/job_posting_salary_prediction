import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder

# --------------------------------------
# Load and split data
# --------------------------------------
_df_jobs_clean = pd.read_csv('data/jobs_clean_nt.csv')
df_X = _df_jobs_clean.drop(columns=["normalized_salary_log10", "normalized_salary", "min_salary", "max_salary", "med_salary"])
df_y = _df_jobs_clean["normalized_salary"]


# --------------------------------------
# Functions for train and test datasets
# --------------------------------------
def train_evaluate_model(model: BaseEstimator):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model, df_X, df_y, cv=cv, 
        return_train_score=True,
        return_estimator=True,
        scoring=["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
    )
    
    train_r2 = scores["train_r2"].mean()
    test_r2 = scores["test_r2"].mean()
    print(f"Mean CV train R2: {train_r2:.4f}")
    print(f"Mean CV test R2: {test_r2:.4f}")

    train_rmse = np.sqrt(-scores["train_neg_mean_squared_error"].mean())
    test_rmse = np.sqrt(-scores["test_neg_mean_squared_error"].mean())
    print(f"Mean CV train RMSE: {train_rmse:.4f}")
    print(f"Mean CV test RMSE: {test_rmse:.4f}")

    train_mae = -scores["train_neg_mean_absolute_error"].mean()
    test_mae = -scores["test_neg_mean_absolute_error"].mean()
    print(f"Mean CV train MAE: {train_mae:.4f}")
    print(f"Mean CV test MAE: {test_mae:.4f}")
    print(f'On average, our predicted salaries are ${test_mae:.2f} off the true salaries')

    # Return the best model based on test R2
    best_estimator = scores['estimator'][np.argmax(scores['test_r2'])]
    return best_estimator


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
