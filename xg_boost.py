

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor

from ocean_forest.config import settings

from ocean_forest.random_forest import load, clean_data, dump_model, load_model


def regress(df=None, env="pp-mattei", random_state=None, depths=False, **kw):
    # evaluate random forest ensemble for regression
    # https://machinelearningmastery.com/random-forest-ensemble-in-python/
    settings.setenv(env=env)
    if df is None:
        print("load dataframe")
        df = load(env=env)
    else:
        df = df.copy(deep=True)
    X,y = clean_data(df, env=env, depths=depths)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    #Set hyper parameters
    rfkw = settings.get("xgb_params", {})
    for key in kw:
        rfkw[key] = kw[key]
    model = XGBRegressor(**rfkw)
    model.env = env
    model.fit(X_train, y_train)
    model.X_test = X_test
    model.y_test = y_test
    model.X_train = X_train
    model.y_train = y_train
    print(r'R2 train: %.3f' % (model.score(model.X_train, model.y_train)))
    print(r'R2 test:  %.3f' % (model.score(model.X_test,  model.y_test)))
    return model
