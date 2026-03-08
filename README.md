# Universal ML template

## Run
pip install -r requirements.txt
python main.py


## House Prices quick setup
In `config.py`:
DATA_CONFIG["target"] = "SalePrice"
DATA_CONFIG["id_col"] = "Id"

TRAIN_CONFIG["task"] = "regression"
TRAIN_CONFIG["metric"] = "rmse"
TRAIN_CONFIG["direction"] = "min"
TRAIN_CONFIG["target_transform"] = "log1p"
TRAIN_CONFIG["prediction_column"] = "SalePrice"


## Titanic
In `config.py`:
DATA_CONFIG["target"] = "Survived"
DATA_CONFIG["id_col"] = "PassengerId"

TRAIN_CONFIG["task"] = "classification"
TRAIN_CONFIG["metric"] = "accuracy"
TRAIN_CONFIG["direction"] = "max"
TRAIN_CONFIG["target_transform"] = None
TRAIN_CONFIG["prediction_column"] = "Survived"

