

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RepeatedKFold
import cudf
from cuml import RandomForestRegressor

from ocean_forest.config import settings

from ocean_forest.random_forest import load, clean_data, dump_model, load_model


def fit(df=None, env="ep-rf", random_state=None, depths=True, xydict=None, **kw):
    # evaluate random forest ensemble for regression
    # https://machinelearningmastery.com/random-forest-ensemble-in-python/
    settings.setenv(env=env)
    if df is None:
        print("load dataframe")
        df = load(env=env)
    else:
        df = df.copy(deep=True)
    if xydict is None:
        X,y = clean_data(df, env=env, depths=depths)
        X = cudf.from_dataframe(X, allow_copy=True) #.to_cupy()
        y = cudf.from_pandas(y) #.to_cupy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.05, random_state=random_state)
    else:
        X_train = cudf.from_dataframe(xydict["X_train"], allow_copy=True)
        X_test  = cudf.from_dataframe(xydict["X_test"], allow_copy=True)
        y_train = cudf.from_pandas(xydict["y_train"])
        y_test  = cudf.from_pandas(xydict["y_test"])


    #Set hyper parametersmv
    rfkw = settings.get("rf_params", {})
    for key in kw:
        rfkw[key] = kw[key]
    model = RandomForestRegressor(**rfkw)
    model.env = env
    model.fit(X_train, y_train)
    model.X_test = X_test.to_pandas()
    model.y_test = y_test.to_pandas()
    model.X_train = X_train.to_pandas()
    model.y_train = y_train.to_pandas()
    print(r'R2 train: %.3f' % (model.score(model.X_train, model.y_train)))
    print(r'R2 test:  %.3f' % (model.score(model.X_test,  model.y_test)))
    return model
