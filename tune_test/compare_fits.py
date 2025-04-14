
import numpy as np
import pandas as pd
import xarray as xr


from sklearn.model_selection import train_test_split,cross_val_score

from ocean_forest.random_forest import load, clean_data
from ocean_forest import random_forest

import model_to_field
import rapids

def data(env="ep_rf"):
    df = load(env=env)
    X,y = clean_data(df, env=env, depths=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)
    return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

def predict(model, dtm1="2012-01-01", dtm2="2012-12-31", depth="Zeu"):
    dtmvec = pd.date_range(dtm1, dtm2, freq="MS")
    dalist = []
    for dtm in dtmvec:
        print(dtm)
        dalist.append(model_to_field.predict_ep_parallel(model, dtm=dtm, depth=depth))
    return xr.concat(dalist, dim="time")

def fit(depth="Zeu"):
    xydict = data()
    cumodel = rapids.fit(xydict=xydict)
    rfmodel = random_forest.regress(xydict=xydict)

    dacu = model_to_field.predict_ep_parallel(cumodel, dtm="2010-07-01", depth="Zeu")
    darf = model_to_field.predict_ep_parallel(rfmodel, dtm="2010-07-01", depth="Zeu")

    dtmvec = pd.date_range("2012-01-01", "2012-12-31", freq="MS")


def diff(model1, model2):

    da = model_to_field.predict_ep_parallel(cumodel, depth="Zeu")
    plt.pcolormesh(np.squeeze(dacu-darf), cmap=cm.RdBu_r,vmin=-50,vmax=50)
