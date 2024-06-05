import mlflow

mlflow.projects.run(
    uri='./',
    entry_point='main',
    experiment_name='salary_model',
    env_manager='local'
)

# In command line
# $ mlflow run . --env-manager local --entry-point main --experiment-name salary_model  # noqa
