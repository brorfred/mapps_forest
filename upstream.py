
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import pylab as pl

import ocean_forest
import match_sat_data 

def match(days=1):
    df = pd.read_hdf("indata/mapps_daily_sat.h5")
    df_m = df.set_index(df.index - pd.DateOffset(days=days))
    df_m = df_m[df_m.index>="1998-01-01"]
    match_sat_data.match_daily(df_m)
    df_m = df_m.set_index(df.index + pd.DateOffset(days=days))
    return df_m

def all():
    for day in range(15,31):
        df = match(days=day)
        df.to_csv(f"mapps_sat_neg_{day:02}_days.csv")
        

def cv(day=1, depths=False, param="pbmax"):
    env = "mapps-alpha-daily" if param=="pbmax" else "mapps-pbmax-daily"
    df = pd.read_csv(f"indata/lagged_data/mapps_sat_neg_{day:02}_days.csv")
    model = ocean_forest.regress(df=df, env="mapps-alpha-daily", depths=depths)
    y = pd.concat((model.y_train, model.y_test))
    X = pd.concat((model.X_train, model.X_test))
    scores = cross_val_score(model, X, y, cv=10)
    return np.median(scores)

def cvplot(alpha, pbmax):
    pl.clf()
    pl.plot(range(1,20), alpha, label="alpha")
    pl.plot(range(1,20), pbmax, label="pBmax")
    pl.ylim(0,1)
    pl.ylabel("R$^2$")
    pl.xlabel("Days lagged")
    pl.legend()

