import sys 
from contextlib import redirect_stdout


import numpy as np
import optuna

import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
#from cuml.ensemble import RandomForestRegressor
import xgboost as xgb



from ocean_forest.random_forest import load, clean_data, dump_model, load_model
import xg_boost


def rf_objective(trial, env="default"):
    """
    
    Ref
    ---
    https://pub.aimind.so/hyperparameter-optimization-of-random-forest-model-using-optuna-for-a-regression-problem-6f49d9b520b7
    https://medium.com/@kalpit.sharma/mastering-random-forest-hyperparameter-tuning-for-enhanced-machine-learning-models-2d1a8c6c426f
    """

    df = load(env)
    X,y = clean_data(df, env=env, depths=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25)


    # Suggest values for hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 10, 500, log=True)
    max_depth = trial.suggest_int("max_depth", 2, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 32)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 32)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    #criterion = trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'friedman_mse'])



    # Create and fit random forest model
    model = RandomForestRegressor(
        n_jobs=-1,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        #criterion=criterion
        #random_state=42,
    )
    model.fit(train_x, train_y)

    # Make predictions and calculate RMSE
    pred_y = model.predict(test_x)
    #rmse = np.sqrt(mean_squared_error(test_y, pred_y))
    #mae = mean_absolute_error(y_test, y_pred)
    r2 = sklearn.metrics.r2_score(test_y, pred_y)

    # Return MAE
    return r2



def xgb_objective(trial, env="ep-xgboost"):
    """
    https://randomrealizations.com/posts/xgboost-parameter-tuning-with-optuna/
    https://github.com/mcb00/ds-templates/blob/main/xgboost-regression.ipynb
    https://www.kaggle.com/code/alisultanov/regression-xgboost-optuna
    https://medium.com/optuna/using-optuna-to-optimize-xgboost-hyperparameters-63bfcdfd3407
    """
    df = load(env=env)
    X,y = clean_data(df, env=env, depths=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)

    param = {
        "nthread" : -1,
        'learning_rate': 0.8,
        #"silent": 1,
        #"tree_method": trial.suggest_categorical("tree_method", ["approx", "hist"]),
        'tree_method': trial.suggest_categorical('tree_method', ['approx', 'hist']),
        #"tree_method": "hist",
        "min_child_weight" : trial.suggest_int("min_child_weight", 1, 250),
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 25, log=True),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 12)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-rmse")
    bst = xgb.train(param, dtrain, evals=[(dtest, "validation")])#, callbacks=[pruning_callback])
    pred_y = bst.predict(dtest)
    return sklearn.metrics.r2_score(test_y, pred_y)

def run():
    study = optuna.create_study(direction="maximize")
    with open('log.txt', 'w') as f:
        with redirect_stdout(f):
            study.optimize(rf_objective, n_trials=500, timeout=600, show_progress_bar=True)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    return study
