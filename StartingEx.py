# Import MLflow
import mlflow


experiment_name = "Unicorn Model"

# Create new experiment
mlflow.create_experiment(experiment_name)

# Tag new experiment
mlflow.set_experiment_tag("version", "1.0")

# Set the experiment
mlflow.set_experiment(experiment_name)

# Find the id of the experiment to delete
exp_data = mlflow.search_experiments(
    filter_string="name like '%Unicorn Model%'")
print(f'Experiment {experiment_name}:')
print(exp_data)

exp_id = exp_data[0].experiment_id
print(f'Experiment Id: {exp_id}')

# Delete Experiments
mlflow.delete_experiment(exp_id)
print(f'Experiment {experiment_name} deleted!')
