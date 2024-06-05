# Import libraries
import mlflow
import pandas as pd
import os
import shutil

from environment import prepare_environment, print, SEED
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Cleaning to allow repetitive running
prepare_environment()
experiment_name = '50_Startups'
saved_dir = 'local-saved-models'
if os.path.exists(f'{saved_dir}/{experiment_name}'):
    shutil.rmtree(f'{saved_dir}/{experiment_name}')
experiments = mlflow.search_experiments(
    filter_string=f"name = '{experiment_name}'")
if experiments:
    for exp_data in experiments:
        exp_id = exp_data.experiment_id
        mlflow.delete_experiment(exp_id)
    print('Experiment cleaned!')


# Reading data
data = pd.read_csv('data/50_Startups.csv')
X = data[['R&D Spend', 'Administration', 'Marketing Spend']]
y = data['Profit']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# Set the experiment
experiment = mlflow.set_experiment(experiment_name)
print(f'Experiment ID: {experiment.experiment_id}')

# Set Auto logging for Scikit-learn flavor
mlflow.sklearn.autolog()

lr = LinearRegression()
lr.fit(X_train, y_train)

# Get a prediction from test data
print(lr.predict(X_test.iloc[[5]]))

# Saving the model
mlflow.sklearn.save_model(lr, f'{saved_dir}/{experiment_name}')
