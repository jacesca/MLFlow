import mlflow
import logging


# Step 0: Prepare the environment
logging.getLogger("mlflow").setLevel(logging.ERROR)  # logging.DEBUG
experiment_name = 'insurance_model'
experiments = mlflow.search_experiments(
    filter_string=f"name = '{experiment_name}'")
if experiments:
    for exp_data in experiments:
        exp_id = exp_data.experiment_id
        mlflow.delete_experiment(exp_id)
    print('Experiment cleaned!')

# Step 1: Training the model
step1 = mlflow.projects.run(
    uri='./',
    entry_point='model_engineering',
    experiment_name=experiment_name,
    env_manager='local',
    parameters={
        'n_jobs': 2,
        'fit_intercept': False
    }
)

# Set Run ID of model training to be passed to Model Evaluation step
run_id = step1.run_id
print(f'Run Id (2): {run_id}')

# Step 2: Evaluating the model
step2 = mlflow.projects.run(
    uri="./",
    entry_point="model_evaluation",
    parameters={
        "run_id": run_id,
    },
    env_manager="local"
)

print(step2.get_status())
