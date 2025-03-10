

import numpy as np
from sklearn.model_selection import KFold
import cudf

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

from ocean_forest.random_forest import clean_data
from ocean_forest.random_forest import load as load_data


def run(model):
    env = model.env
    df = load_data(env=env)
    X,y = clean_data(df, env=env, depths=True)
    X = cudf.from_dataframe(X, allow_copy=True) #.to_cupy()
    y = cudf.from_pandas(y) #.to_cupy()

    n_splits = 10
    cv = KFold(n_splits=n_splits,  shuffle=True)
    scores = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1)
    print('Accuracy: %.3f ,\nStandard Deviations :%.3f' %
          (np.mean(scores), np.std(scores)))
    return scores

def run_rf(model):
    env = model.env
    df = load_data(env=env)
    X,y = clean_data(df, env=env, depths=True)
    #X = cudf.from_dataframe(X, allow_copy=True) #.to_cupy()
    #y = cudf.from_pandas(y) #.to_cupy()

    n_splits = 10
    cv = KFold(n_splits=n_splits,  shuffle=True)
    scores = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1)
    print('Accuracy: %.3f ,\nStandard Deviations :%.3f' %
          (np.mean(scores), np.std(scores)))
    return scores