# Import needed libraries
import mlflow
from mlflow import MlflowClient
from environment import print


# Create an instance
client = MlflowClient()

# Print the object
print('Client:', client)

##########################################################
# Search for registered models
##########################################################
filter_string = "name LIKE '%Tree%'"
result = client.search_registered_models(filter_string=filter_string)
print(result)

##########################################################
# Transitioning models
##########################################################
client.transition_model_version_stage(
    name="KNeighborsClassifier()",
    version=1,
    stage="Production"
)
filter_string = "name LIKE '%KNeighbors%'"
result = client.search_registered_models(filter_string=filter_string)
print(result)


##########################################################
# Load models
##########################################################
# Follow format "models:/model_name/version" or "models:/model_name/stage"
model = mlflow.sklearn.load_model('models:/KNeighborsClassifier()/1')
print(model)

# Serving models: using cli
# $ mlflow models serve -m "models:/KNeighborsClassifier()/Production"
# It requires pyenv binary.
# See https://github.com/pyenv-win/pyenv-win#installation
# for installation instructions.
