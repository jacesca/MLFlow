import mlflow

mlflow.projects.run(
    uri='./',
    entry_point='main',
    experiment_name='insurance_model',
    env_manager='local',
    parameters={
        'n_jobs_param': 2,
        'intercept_param': False
    }
)

# In command line
# $ mlflow run . --env-manager local --entry-point main --experiment-name insurance_model -P n_jobs_param=3, -P intercept_param=False  # noqa
