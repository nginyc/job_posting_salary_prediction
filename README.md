# Job Posting Salary Prediction

## Setup

```sh
pyenv install
pyenv exec python -m venv ./venv
source ./venv/bin/activate
```

```sh
pip install -r requirements.txt
```

## Run notebooks

Run the notebooks in the following order:

1. `export_dataset.ipynb` reads original dataset from Kaggle and exports the raw dataset to `data/jobs.csv`
2. `clean_datset.ipynb` reads from `data/jobs.csv`, filters, cleans and pre-processes the dataset for our problem statement (before a train-test split), and exports the clean dataset for training to `data/jobs_clean.csv`
3. `normalize_job_titles.ipynb` reads from `data/jobs_clean.csv`, experiments and trains a model for normalizing job titles, annotates the dataset with a new column `norm_title`, and exports the annotated dataset `data/jobs_clean_nt.csv`
4. The following other notebooks use `data/jobs_clean_nt.csv` to train and evaluate different models:
    - `evaluate_stats_models.ipynb`
    - `train_linear_model.ipynb`
    - `train_xgb_model.ipynb`


