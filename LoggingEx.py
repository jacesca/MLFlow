import mlflow

from environment import prepare_environment, print, SEED
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Cleaning to allow repetitive running
experiment_name = 'LR Experiment'
prepare_environment()
exp_data = mlflow.search_experiments(
    filter_string=f"name = '{experiment_name}'")
if exp_data:
    exp_id = exp_data[0].experiment_id
    mlflow.delete_experiment(exp_id)
    print('Experiment cleaned!')


# Global functions
def read_data():
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    return X, y, X_train, y_train, X_test, y_test


def run_model(X_train, y_train, X_test, y_test):
    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Predict on the test set
    y_pred = lr.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    return lr, params, y_pred, accuracy


# Set experiment
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
experiment = mlflow.set_experiment(experiment_name)
print(f'Experiment ID: {experiment.experiment_id}')

# Start a run
run = mlflow.start_run()

# Reading the data and run the model
X, y, X_train, y_train, X_test, y_test = read_data()
model, params, y_pred, score = run_model(X_train, y_train, X_test, y_test)

# Log a metric
mlflow.log_metric("score", score)

# Log a parameter
mlflow.log_params(params)

# Log an artifact
mlflow.log_artifact(__file__)

print('Model Tracked!')
print(run.info)

mlflow.end_run()
