import os
import calendar

import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import cm
from pyresample import kd_tree
from joblib import load as joblib_load
from joblib import Parallel, delayed
import projmap
import pylab as pl

from pararegress import linregress
from model_to_field import load_model, open_dataset, ds_to_df

def predict_ep_all_depths(model, dtm="2010-01-01", ds=None, n_jobs=10):
    """Predict monthly EP globally using a trained RF model - multicore"""
    ds = open_dataset(dtm=dtm) if ds is None else ds

    def setup_dataframe():
        df = ds_to_df(ds)
        for key in ["chl", "pprod", "pp_obs", "ep_obs"]:
            if key in df:
                df[key] = np.log(df[key])
        return df[model.X_train.keys()]

    def predict():
        dflist = np.array_split(df,10)
        parallel = Parallel(n_jobs=n_jobs)
        ep_pred = parallel(delayed(model.predict)(dflist[i]) for i in range(n_jobs))
        da = ds.lats * 0
        da = da.rename("export_production")
        da.data[ds.nanmask.data] = np.hstack(ep_pred)
        return da

    print("Read dataframe")
    df = setup_dataframe()
    eplist = []
    zvec = np.arange(100,510,10)
    for zlev in zvec:
        df["depth"] = zlev
        ep = predict()
        eplist.append(ep[0,:,:])
    da = xr.concat(eplist, dim="depth")
    da = da.assign_coords({"depth":zvec})
    da.attrs["env"] = getattr(model, 'env', 'default')
    return da

def year(year=2010, model=None):
    """Calculate all monthly global EP fields for a year"""
    model  = load_model() if model is None else model
    dtmvec = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    eplist = []
    for dtm in dtmvec:
        print(dtm)
        eplist.append(predict_ep(model, dtm))
    ds = xr.concat(eplist, dim="time")
    ds.attrs["env"] = getattr(model, 'env', 'default')
    return ds.to_dataset()

def calc_b(epz, z, z0):
    """Calculate b using a linear regression

    ref
    ---
    Britten et al Eq 3, doi:10.3389/fenvs.2021.491636
    """
    xvec = np.log(z/z0)
    yvec = np.log(epz)
    regr = linregress(xvec, yvec)
    return regr.slope

def calc_b_for_field(da=None, dtm="2010-01-01", model=None, z0=100):
    model = load_model() if model is None else model
    da = predict_ep_all_depths(model=model, dtm=dtm) if da is None else da
    xvec = np.log(da.depth.data/z0)
    regr = linregress(np.squeeze(da.data), xvec)
    ds =  xr.Dataset({"b_param":(("time","lat","lon"), regr.slope[None,:,:]),
                    f"EP_{z0}":(("time","lat","lon"), regr.intercept[None,:,:])},
                    coords={"time":[da.time.data], "lat":da.lat, "lon":da.lon})
    return ds

def b_for_a_year(year=2010, model=None):
    model = load_model() if model is None else model
    dtmvec = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    for dtm in dtmvec:
        fn = f"ncfiles/b_fit_{dtm.year}{dtm.month:02}_{model.env}.nc"
        if os.path.isfile(fn):
            continue
        print(dtm, fn)
        ds = calc_b_for_field(model=model, dtm=dtm)
        ds.to_netcdf(fn)
        del ds

def b_map(year=2010, month=1):
    ds = xr.open_dataset(f"ncfiles/b_fit_{year}{month:02}_no_regions.nc")
    ds.b_param.data[ds.b_param.data==0] = np.nan
    mp = projmap.Map("glob")
    pl.clf()
    fig = pl.gcf()
    #fig.patch.set_facecolor('blue')
    fig.patch.set_alpha(0)
    mp.pcolor(ds.lon, ds.lat, np.squeeze(ds.b_param), vmin=-1.4, vmax=-0.2)
    mp.nice()
    mp.colorbar()
    pl.suptitle(f"{calendar.month_abbr[month]} {year}", x=0.5, y=0.8)
    pl.savefig(f"figs/b_maps/b_map_{year}-{month:02}.png",
        dpi=400, bbox_inches="tight", transparent=True)
    pl.close("all")
