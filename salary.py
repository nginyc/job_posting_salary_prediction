from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import functools
from sentence_transformers import SentenceTransformer

# --------------------------------------
# Functions for train and test datasets
# --------------------------------------

SALARY_COLUMNS = [
    "normalized_salary", "min_salary", "max_salary", "med_salary",
    "extracted_normalized_salary", "extracted_min_salary", "extracted_max_salary", "extracted_salary"
]

@functools.lru_cache(maxsize=None)
def get_train_dataset(include_extracted_salaries=False):
    df = pd.read_csv('data/jobs_train.csv')

    salaries = df['normalized_salary']
    if include_extracted_salaries:
        salaries = salaries.fillna(df['extracted_normalized_salary'])

    df = df[salaries.notna()]
    salaries = salaries[salaries.notna()]

    X = df.drop(columns=SALARY_COLUMNS)
    y = list(salaries.values)
    
    return X, y

@functools.lru_cache(maxsize=None)
def get_test_dataset():
    df = pd.read_csv('data/jobs_test.csv')
    X = df.drop(columns=SALARY_COLUMNS)
    y = list(df["normalized_salary"].values)
    return X, y

def evaluate_train_predictions(y_train_pred, y_train):
    '''
    Evaluate the train predictions against the true train values (normalized_salary)
    '''
    print(f"Train size: {len(y_train)}")

    train_r2 = r2_score(y_train, y_train_pred)
    print(f"Train R2: {train_r2:.4f}")

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"Train RMSE: {train_rmse:.4f}")

    train_mae = mean_absolute_error(y_train, y_train_pred)
    print(f"Train MAE: {train_mae:.4f}")

    result = { 
        'r2': train_r2,
        'rmse': train_rmse,
        'mae': train_mae    
    }

    return result

def evaluate_test_predictions(y_test_pred):
    '''
    Evaluate the test predictions against the true test values (normalized_salary)
    '''
    _, y_test = get_test_dataset()
    print(f"Test size: {len(y_test)}")

    test_r2 = r2_score(y_test, y_test_pred)
    print(f"Test R2: {test_r2:.4f}")

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(f"Test RMSE: {test_rmse:.4f}")

    test_mae = mean_absolute_error(y_test, y_test_pred)
    print(f"Test MAE: {test_mae:.4f}")

    result = { 
        'r2': test_r2,
        'rmse': test_rmse,
        'mae': test_mae
    }

    return result

## --------------------------------------
## Functions to encode features
## --------------------------------------

@functools.lru_cache(maxsize=None)
def get_sentence_transfomer():
    # We use the pre-trained sentence-BERT model for encoding text data
    # We pick the relatively smaller 80MB model with highest performance on sentence embeddings
    # See https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#original-models
    return SentenceTransformer('all-MiniLM-L6-v2')

class SentenceBertEncoder(BaseEstimator):  
    def __init__(self):
        self._model = get_sentence_transfomer()

    def fit(self, X, y=None):
        return self
    
    def encode(self, X):
        return self._model.encode(list(X))

    def transform(self, X: pd.DataFrame):
        return self.encode(np.array(X).flatten()).reshape(X.shape[0], -1)
    
    def similarity(self, X, Y):
        return self._model.similarity(X, Y)
    
    def get_feature_names_out(self, input_features: Optional[list[str]] = None):
        d = self._model.get_sentence_embedding_dimension() or 0
        names = list(input_features) if input_features is not None and len(input_features) > 0 else []
        return [f'{name}_sbert_{i}' for name in names for i in range(d) ]

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

## --------------------------------------
## Preprocessors
## --------------------------------------

def get_preprocessor() -> Pipeline:
    return make_pipeline(
        ColumnTransformer(
            transformers=[
                ('title_sbert_pca_encoder', make_pipeline(
                    SentenceBertEncoder(),
                ), ['title']),
                ('location_sbert_pca_encoder', make_pipeline(
                    SentenceBertEncoder(),
                ), ['location']),
                ('company_industries_sbert_pca_encoder', make_pipeline(
                    SentenceBertEncoder(),
                ), ['company_industries']),
                ('skills_sbert_pca_encoder', make_pipeline(
                    SimpleImputer(strategy='constant', fill_value='Unknown'),
                    SentenceBertEncoder(),
                ), ['extracted_skill_requirement']),
                ('education_sbert_pca_encoder', make_pipeline(
                    SimpleImputer(strategy='constant', fill_value='Unknown'),
                    SentenceBertEncoder(),
                ), ['extracted_education_requirement']),
                ('certification_sbert_pca_encoder', make_pipeline(
                    SimpleImputer(strategy='constant', fill_value='Unknown'),
                    SentenceBertEncoder(),
                ), ['extracted_certification_requirement']),
                ('experience_sbert_pca_encoder', make_pipeline(
                    SimpleImputer(strategy='constant', fill_value='Unknown'),
                    SentenceBertEncoder(),
                ), ['extracted_experience_requirement']),
                ('text_one_hot_encoder', 
                    OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist', min_frequency=10), 
                    ['title', 'location', 'company_industries', 'company_country', 'company_state']
                ),
                ('enum_one_hot_encoder', OneHotEncoder(sparse_output=False, handle_unknown='error'), 
                    ['formatted_experience_level', 'formatted_work_type']
                ),
                ('experience_level', experience_level_encoder, ['formatted_experience_level']),
                ('work_type', work_type_encoder, ['formatted_work_type']),
                ('remote_allowed', 'passthrough', ['remote_allowed']),
                ('company_employee_count', make_pipeline(
                    SimpleImputer(strategy='median'),
                ), ['company_employee_count']),
            ],
            remainder='drop'
        ),
        StandardScaler(),
    )