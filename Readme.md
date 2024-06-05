# ML Experiment tracking
A sample code on how to use MLFlow

Features:
- 

## Quick start
1. Run the MLFlow Service
```
mlflow server --host 127.0.0.1 --port 5080

# or

mlflow ui
```

2. Train a model and prepare metadata for logging 
```
python simpleML.py        # MLflow Quickstart
python anotherSample.py   # MLflow RandomForestClassifier
```

3. View the Run in the MLflow UI
> Go to [http://localhost:8080](http://localhost:5080)

## Installing using GitHub
- Fork the project into your GitHub
- Clone it into your dektop
```
git clone https://github.com/jacesca/MLFlow.git
```
- Setup environment (it requires python3)
```
python -m venv venv
source venv/bin/activate  # for Unix-based system
venv\Scripts\activate  # for Windows
```
- Install requirements
```
pip install -r requirements.txt
```

## Extra documentation
- [MLFlow](https://mlflow.org/)
- [Quick Start - MLFlow](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html)
- [SHAP documentation](shap.readthedocs.io)
- [Logging levels](https://docs.python.org/3/library/logging.html#levels)
- [MLFlow RestAPI](https://mlflow.org/docs/latest/rest-api.html#create-experiment)
