

import numpy as np
import pandas as pd
import pylab as pl

from ocean_forest.random_forest import load, clean_data, dump_model, load_model


from joblib import Parallel, delayed

def expand_row_with_all_depths(df, zvec=None):
    zvec = np.arange(0, 505, 5) if zvec is None else zvec
    df = df.iloc[[0]]
    dflist = []
    df2 = pd.DataFrame()
    for zlev in zvec:
        row = df.copy(deep=True)
        row["depth"] = zlev
        dflist.append(row)
    return pd.concat(dflist, axis=0)

def predict_all_depths(model, df, zmin=0, zmax=500, dz=5, zvec=None, n_jobs=10):
    """Predict monthly EP globally using a trained RF model - multicore"""
    #ds = open_dataset(dtm=dtm) if ds is None else ds

    #def setup_dataframe():
    #    df = ds_to_df(ds)
    #    for key in ["chl", "pprod", "pp_obs", "ep_obs"]:
    #        if key in df:
    #            df[key] = np.log(df[key])
    #    return df[model.X_train.keys()]

    def predict(df):
        dflist = np.array_split(df,10)
        parallel = Parallel(n_jobs=n_jobs)
        y_pred = parallel(delayed(model.predict)(dflist[i]) for i in range(n_jobs))
        return np.hstack(y_pred)

    print("Read dataframe")
    X_pred = df[model.X_test.keys()].copy(deep=True).dropna()
    eplist = []
    zvec = np.arange(zmin, zmax+dz, dz) if zvec is None else zvec
    for zlev in zvec:
        X_pred.loc[:,"depth"] = zlev
        ep = predict(X_pred)
        eplist.append(ep)
    return np.array(eplist)



def vertical_curve(model):
    df = load(env="ep_rf")
    zvec = np.arange(5,510,5)
    resid = dict(fit=[], sat=[], obs=[], ep_obs=[], b_obs=[], b_sat=[])
    for ugrname,ugr in df.groupby("UUID"):
        for grname,gr in ugr.groupby("start_time"):
            if (len(np.unique(gr.depth)) > 2) and np.isfinite(gr.iloc[0]["ep_sat"]):
                Xprod = zresolved_dataframe(gr[model.X_train.keys()])
                ep_sat = np.exp(model.predict(Xprod))
                b_sat,ep100_sat2 = calc_b(ep_sat[5:], Xprod.depth[5:], 100, True)
                ep100_sat = np.log(ep_sat[np.nonzero(Xprod.depth.values==110)[0][0]])
                print(np.exp(ep100_sat),np.exp(ep100_sat2))
                #b_sat,ep100_sat = calc_b(gr.ep_sat, gr.depth, 100, True)
                ep_mrt_sat = np.exp(ep100_sat) * (Xprod.depth/100)**(b_sat)

                b_obs,ep100_obs = calc_b(gr.ep_obs, gr.depth, 100, True)
                ep_mrt_obs = np.exp(ep100_obs) * (Xprod.depth/100)**(-1.46)
                #print(ugrname,grname)
                pl.clf()
                pl.plot(ep_sat, -Xprod.depth, lw=1, label="ep_sat")
                pl.plot(ep_mrt_sat, -Xprod.depth, lw=1, label="Fit to ep_sat")
                pl.scatter(gr.ep_sat, -gr.depth, 5)
                pl.scatter(gr.ep_obs, -gr.depth, 5, label="ep_obs", c="tab:green")
                pl.plot(ep_mrt_obs, -Xprod.depth, lw=1, label="Fit to ep_obs")

                pl.ylim(-500,0)
                pl.xlim(0,150)
                pl.legend(loc='lower right')
                pl.savefig(f"figs/vertical_curves/vertical_curve_{ugrname}_{str(grname.date())}_bcurve.pdf")

                ep_mrt_obs = np.exp(ep100_obs) * (gr.depth/100)**(-1.46)
                ep_mrt_sat = np.exp(ep100_sat) * (gr.depth/100)**(b_sat)
                resid["ep_obs"].append(gr.ep_obs)
                resid["obs"].append((gr.ep_obs - ep_mrt_obs).values)
                resid["fit"].append((gr.ep_obs - ep_mrt_sat).values)
                resid["sat"].append((gr.ep_obs - gr.ep_sat).values)
                resid["b_sat"].append(b_sat)
                resid["b_obs"].append(b_obs)

    for key in resid:
        resid[key] = np.hstack(resid[key])
    print(f"MAE sat:  {np.mean(np.abs(resid['sat'])):.2f}")
    print(f"MAE fit:  {np.mean(np.abs(resid['fit'])):.2f}")
    print(f"MAE obs:  {np.mean(np.abs(resid['obs'])):.2f}")
    print(f"MAPE sat: {np.mean(np.abs(resid['sat']/resid['ep_obs'])):.2}")
    print(f"MAPE fit: {np.mean(np.abs(resid['fit']/resid['ep_obs'])):.2}")
    print(f"MAPE obs: {np.mean(np.abs(resid['obs']/resid['ep_obs'])):.2}")
    return resid

def vertical_scatter():
    df = load()
    for grname,gr in df.groupby("UUID"):
        if len(np.unique(gr.depth)) > 2:
            print(grname)
            pl.clf()
            pl.scatter(gr.ep_obs, -gr.depth, 5, label="ep_obs")
            pl.scatter(gr.ep_sat, -gr.depth, 5, label="ep_sat")
            pl.ylim(-500,0)
            pl.xlim(0,150)

            pl.legend()
            pl.savefig(f"figs/vertical_scatter_{grname}.png")

def depths(model):
    zvec = np.arange(5,505,5)
    X = pd.concat([model.X_test, model.X_train]).copy()
    y = np.exp(pd.concat([model.y_test, model.y_train]).copy())

    epavg = []
    epstd = []
    epmat = []
    for depth in zvec:
        X.depth = depth
        yvec = np.exp(model.predict(X))
        epavg.append(np.median(yvec))
        epstd.append(np.std(yvec))
        epmat.append(yvec)
    epmat = np.array(epmat)
    epavg = np.array(epavg)
    epstd = np.array(epstd)
    return epmat
