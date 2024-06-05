import mlflow

from environment import prepare_environment, print, SEED
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from mlflow.models import infer_signature


# Cleaning to allow repetitive running
experiment_name = 'Iris Experiment'
prepare_environment()
experiments = mlflow.search_experiments(
    filter_string=f"name = '{experiment_name}'")
if experiments:
    for exp_data in experiments:
        exp_id = exp_data.experiment_id
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


# Reading the data
X, y, X_train, y_train, X_test, y_test = read_data()

# Set experiment
experiment = mlflow.set_experiment(experiment_name)
print(f'Experiment ID: {experiment.experiment_id}')

# Training the model
models_to_train = [LogisticRegression(), KNeighborsClassifier(),
                   DecisionTreeClassifier()]
for model in models_to_train:
    # Start a run
    run = mlflow.start_run(run_name=str(model))

    # Train the model
    model.fit(X_train, y_train)

    # Calculate metrics
    score = model.score(X_test, y_test)

    # Log a metric
    mlflow.log_metric("score", score)

    # Log a parameter
    mlflow.log_params(model.get_params())

    # Infer the model signature
    signature = infer_signature(X, model.predict(X))

    # Log a model
    mlflow.sklearn.log_model(
        sk_model=model,
        # The convention set artifact_path='model',
        artifact_path='model',
        signature=signature,
        input_example=X,
        registered_model_name=str(model),
    )

    # Log an artifact
    mlflow.log_artifact(__file__)

    # End run
    mlflow.end_run()

# # Searching runs
# print('MLFlow Runs:', mlflow.search_runs())

# Search runs example
score_filter = "metrics.score > 0.60"
df = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=score_filter,
        order_by=["metrics.score DESC"]
    )
print('DF info:', df.info())
print(
    "Filter search result:",
    df[['experiment_id', 'run_id', 'tags.mlflow.runName',
       'status', 'metrics.score']]
)
