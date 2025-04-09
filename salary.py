from typing import Optional
from idna import encode
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
import functools
import seaborn as sns
import matplotlib.pyplot as plt
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
    (_, _, _, df_y_test) = get_dataset()
    y_test = df_y_test.values

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

def plot_evaluation_results(names, results, x_label_rotation=0):
    '''
    Plot the test evaluation results for each model.
    '''

    # Performance metrics for each model
    metrics_data = {
        'Model': names,
        'Test R²': [result['r2'] for result in results],
        'Test RMSE': [result['rmse'] for result in results],
        'Test MAE': [result['mae'] for result in results]
    }

    metrics_df = pd.DataFrame(metrics_data)

    # Create subplots for Test and Train metrics comparison with transparent background
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='none')

    # Plot Test R² Comparison
    sns.barplot(x='Model', y='Test R²', data=metrics_df, ax=axes[0], palette='Blues')
    axes[0].set_title('Test R² Comparison', fontsize=14)
    axes[0].set_ylabel('Test R²', fontsize=12)
    axes[0].set_xlabel(None)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=x_label_rotation)

    # Plot Test RMSE Comparison
    sns.barplot(x='Model', y='Test RMSE', data=metrics_df, ax=axes[1], palette='Greens')
    axes[1].set_title('Test RMSE Comparison', fontsize=14)
    axes[1].set_ylabel('Test RMSE', fontsize=12)
    axes[1].set_xlabel(None)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=x_label_rotation)

    # Plot Test MAE Comparison
    sns.barplot(x='Model', y='Test MAE', data=metrics_df, ax=axes[2], palette='Oranges')
    axes[2].set_title('Test MAE Comparison', fontsize=14)
    axes[2].set_ylabel('Test MAE', fontsize=12)
    axes[2].set_xlabel(None)
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=x_label_rotation)

    # Remove the borders (spines) and background for each subplot
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor('none')  
        ax.grid(False) 

    plt.show()



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
