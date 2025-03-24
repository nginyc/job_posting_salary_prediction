from idna import encode
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
import functools
from sentence_transformers import SentenceTransformer

# --------------------------------------
# Functions for train and test datasets
# --------------------------------------

@functools.lru_cache(maxsize=None)
def get_dataset():
    df_jobs_clean = pd.read_csv('data/jobs_clean_jd.csv')
    df_X = df_jobs_clean.drop(columns=["normalized_salary_log10", "normalized_salary", "min_salary", "max_salary", "med_salary"])
    df_y = df_jobs_clean["normalized_salary"]
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)
    return (df_X_train, df_X_test, df_y_train, df_y_test)

def get_train_dataset():
    (df_X_train, _, df_y_train, _) = get_dataset()
    return df_X_train, df_y_train

def get_test_dataset():
    (_, df_X_test, _, df_y_test) = get_dataset()
    return df_X_test, df_y_test

def evaluate_train_predictions(y_train_pred):
    '''
    Evaluate the train predictions against the true train values (normalized_salary)
    '''
    (_, _, df_y_train, _) = get_dataset()
    y_train = df_y_train.values

    train_r2 = r2_score(y_train, y_train_pred)
    print(f"Train R2: {train_r2:.4f}")

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"Train RMSE: {train_rmse:.4f}")

    train_mae = mean_absolute_error(y_train, y_train_pred)
    print(f"Train MAE: {train_mae:.4f}")

def evaluate_test_predictions(y_test_pred):
    '''
    Evaluate the test predictions against the true test values (normalized_salary)
    '''
    (_, _, df_y_train, df_y_test) = get_dataset()
    y_train_mean = df_y_train.mean()
    y_test = df_y_test.values

    test_r2 = r2_score(y_test, y_test_pred)
    print(f"Test R2: {test_r2:.4f}")

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(f"Test RMSE: {test_rmse:.4f}")

    test_mae = mean_absolute_error(y_test, y_test_pred)
    print(f"Test MAE: {test_mae:.4f}")

    test_mae_mean = mean_absolute_error(y_test, [y_train_mean]*len(y_test))
    improvement = ((test_mae_mean-test_mae) / test_mae_mean)*100
    print(f'On average, our predicted salaries are ${test_mae:.2f} off the true salaries')
    print(f'This is {improvement:.2f}% better than a naive global mean')


## --------------------------------------
## Functions to encode features
## --------------------------------------

class SentenceBertEncoder(BaseEstimator):  
    def __init__(self):
        # We use the pre-trained sentence-BERT model for encoding text data
        # We pick the model with highest performance on sentence embeddings
        # See https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#original-models
        self._model = SentenceTransformer('all-mpnet-base-v2')

    def fit(self, X, y=None):
        return self
    
    def encode(self, X):
        return self._model.encode(list(X))

    def transform(self, X: pd.DataFrame):
        return self.encode(X.T.values[0])
    
    def similarity(self, X, Y):
        return self._model.similarity(X, Y)
    
    def get_feature_names_out(self):
        return [f'sbert_{i}' for i in range(768)]
    

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
