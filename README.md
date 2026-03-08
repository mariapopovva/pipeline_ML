# Universal ML template

## Run
pip install -r requirements.txt <br>
python main.py


## House Prices quick setup
In `config.py`:<br>
DATA_CONFIG["target"] = "SalePrice" <br>
DATA_CONFIG["id_col"] = "Id"<br>

TRAIN_CONFIG["task"] = "regression"<br>
TRAIN_CONFIG["metric"] = "rmse"<br>
TRAIN_CONFIG["direction"] = "min"<br>
TRAIN_CONFIG["target_transform"] = "log1p"<br>
TRAIN_CONFIG["prediction_column"] = "SalePrice"<br>


## Titanic
In `config.py`:<br>
DATA_CONFIG["target"] = "Survived"<br>
DATA_CONFIG["id_col"] = "PassengerId"<br>

TRAIN_CONFIG["task"] = "classification"<br>
TRAIN_CONFIG["metric"] = "accuracy"<br>
TRAIN_CONFIG["direction"] = "max"<br>
TRAIN_CONFIG["target_transform"] = None<br>
TRAIN_CONFIG["prediction_column"] = "Survived"<br>

