import os

import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import cm
from pyresample import kd_tree
from joblib import load as joblib_load
from joblib import Parallel, delayed

from njord import longhurst, modis, ostia, tiho_psd
from njord import oc_cci_local as oc_cci

import ocean_forest
from ocean_forest.random_forest import load as load_data
#from utils import daylength

DATA_VARS = ["kd_490", 'water_class13', 'water_class12', 'water_class10', 'atot_510', 'atot_560']

def sat_filename(dtm="2010-01-01", datadir="/data/forest"):
    dtm = pd.to_datetime(dtm)
    return f"{datadir}/old/sat_fields_{dtm.year}_{dtm.month:02}.nc"

def longhurst_4km():
    ds = longhurst.open_dataset()
    kw = dict(source_geo_def=longhurst.setup_grid(),
              target_geo_def=oc_cci.setup_grid(),
              radius_of_influence=50000)
    regions = kd_tree.resample_nearest(data=ds.regions.data, **kw)
    basins  = kd_tree.resample_nearest(data=ds.basins.data,  **kw)
    biomes  = kd_tree.resample_nearest(data=ds.biomes.data,  **kw)
    return regions,basins,biomes

def ostia_4km(dtm="2010-01-01"):
    ds = ostia.open_dataset(dtm=dtm, timetype="mo")
    kw = dict(source_geo_def=ostia.setup_grid(),
              target_geo_def=oc_cci.setup_grid(),
              radius_of_influence=50000)
    return kd_tree.resample_nearest(data=ds.sst.data, **kw)

regions,basins,biomes = longhurst_4km()

def load_model(datadir="rf_models", filename="ep_model_no_regions_76443.joblib"):
    return joblib_load(os.path.join(datadir, filename))

def open_dataset(dtm="2010-01-01", cache=False):
    """Load xarray datasets with all features needed to predict EP"""
    fn = sat_filename(dtm)
    if os.path.isfile(fn) and cache:
        return xr.open_dataset(fn)
    dtm = pd.to_datetime(dtm) if isinstance(dtm, str) else dtm
    ds = oc_cci.open_dataset(dtm=dtm, data_var="chlor_a", timetype="m")["chlor_a"].to_dataset()
    for key in DATA_VARS:
        ds[key]  = oc_cci.open_dataset(dtm=dtm, data_var=key, timetype="m")[key]
    psd = tiho_psd.open_dataset(dtm=dtm, data_var="PSD_slope")["PSD_slope"].data
    ds["psd"]  = (("time","lat","lon"), psd[None,:,:])
    ds["Zeu"]  = 4.6/ds["kd_490"]
    ds["Zeu01"]  = 6.9/ds["kd_490"]
    ds["sst"]  = (("time","lat","lon"), ostia_4km(dtm=dtm)[None,:,:])
    ds["sat_par"] = (("time","lat","lon"),
        modis.open_dataset(dtm=dtm, timetype="mo")["par"].data[None,:,:])
    lons,lats = np.meshgrid(ds.lon, ds.lat)
    ds["lons"]      = (("time","lat","lon"), lons[None,:,:])
    ds["lats"]      = (("time","lat","lon"), lats[None,:,:])
    ds["month"]     = ds.lats*0+dtm.month
    ds["longhurst"] = (("time","lat","lon"), regions[None,:,:])
    ds["basins"]    = (("time","lat","lon"), basins[None,:,:])
    ds["biomes"]    = (("time","lat","lon"), biomes[None,:,:])
    #ds["daylength"] = (("time","lat","lon"), daylength(dtm.day_of_year, ds["lats"].data))
    nanmask = np.isfinite(ds.chlor_a + ds.kd_490 + ds.Zeu + ds.sst + ds.sat_par + ds.psd)
    ds["nanmask"] = (("time","lat","lon"), nanmask.data)
    return ds

def ds_to_df(ds, depth="Zeu01"):
    """Convert xarray dataset with EP features to pandas dataframe"""
    df = pd.DataFrame({
                       "chl":ds["chlor_a"].data[ds.nanmask.data],
                       "sst":ds["sst"].data[ds.nanmask.data],
                       "psd":ds["psd"].data[ds.nanmask.data],
                       "Zeu":ds["Zeu"].data[ds.nanmask.data],
                       "longhurst":ds["longhurst"].data[ds.nanmask.data],
                       "basin":ds["basins"].data[ds.nanmask.data],
                       "biome":ds["biomes"].data[ds.nanmask.data],
                       "month":ds["month"].data[ds.nanmask.data],
                       "sat_par":ds["sat_par"].data[ds.nanmask.data],
                       "kd_490":ds["kd_490"].data[ds.nanmask.data],
                       "lat":ds["lats"].data[ds.nanmask.data],
                       "lon":ds["lons"].data[ds.nanmask.data],
                       #"daylength":ds["daylength"].data[ds.nanmask.data],
                      })
    for key in DATA_VARS:
        df[key] = ds[key].data[ds.nanmask.data]
    if depth == "Zeu01":
        df["depth"] = ds["Zeu01"].data[ds.nanmask.data]
    elif depth == "Zeu":
        df["depth"] = ds["Zeu"].data[ds.nanmask.data]
    else:
        df["depth"] = depth
    return df

def predict_ep(model, dtm="2010-01-01", depth="Zeu01"):
    """Predict monthly EP globally using a trained RF model"""
    print("Read dataframe")
    ds = open_dataset(dtm=dtm)
    df = ds_to_df(ds, depth=depth)
    print("Predict EP")
    df = df[model.X_train.keys()]
    ep = np.exp(model.predict(df))
    da = ds.lats * 0
    da = da.rename("export_production")
    da.data[ds.nanmask.data] = ep
    da.attrs["env"] = getattr(model, 'env', 'default')
    return da

def predict_ep_parallel(model, dtm="2010-01-01", depth="Zeu01", ds=None, n_jobs=10):
    """Predict monthly EP globally using a trained RF model - multicore"""
    print("Read dataframe")
    ds = open_dataset(dtm=dtm) if ds is None else ds
    df = ds_to_df(ds, depth=depth)
    print("Predict EP")
    df = df[model.X_train.keys()]
    dflist = np.array_split(df,10)
    parallel = Parallel(n_jobs=n_jobs)
    ep_list = parallel(delayed(model.predict)(dflist[i]) for i in range(n_jobs))
    ep_pred = np.exp(np.hstack(ep_list))
    da = ds.lats * 0
    da = da.rename("export_production")
    da.data[ds.nanmask.data] = np.hstack(ep_pred)
    da.attrs["env"] = getattr(model, 'env', 'default')
    da.attrs["depth"] = depth
    return da



def predict_year(year=2010, depth="Zeu01", model=None):
    """Calculate all monthly global EP fields for a year"""
    if model is None:
        model  = load_model()
    dtmvec = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    eplist = []
    for dtm in dtmvec:
        print(dtm)
        eplist.append(predict_ep_parallel(model, dtm, depth=depth))
    ds = xr.concat(eplist, dim="time").to_dataset()
    ds.attrs["env"] = getattr(model, 'env', 'default')
    ds.attrs["depth"] = ds["export_production"].attrs["depth"]
    return ds

def to_netcdf(ds, datadir="ncfiles"):
    year = pd.to_datetime(ds.time).year[0]
    filename = f"{datadir}/EP_{ds.env}_depth-{ds.depth}_{year}.nc"
    ds.to_netcdf(filename)

def climatology(month=1):
    """Calculate EP climatologies"""
    df = load_data.load()
    model  = ocean_forest.regress(df=df)
    dtmvec = pd.date_range(f"1999-{month:02}-01", f"2019-{month:02}-01", freq="12MS")
    eplist = []
    for dtm in dtmvec:
        print(dtm)
        eplist.append(predict_ep(model, dtm))
    return xr.concat(eplist, dim="time")

def all():
    model = joblib_load("rf_models/ep_model_below_zeu_no_regions_depths_73969.joblib")
    for year in range(2003,2023):
        print(year)
        #ds = predict_year(year, model=model, depth="Zeu")
        #to_netcdf(ds)
        #ds = predict_year(year, model=model, depth="Zeu01")
        #to_netcdf(ds)
        ds = predict_year(year, model=model, depth=100)
        to_netcdf(ds)

"""
from njord2 import oc_cci, ifado
from pyresample import kd_tree
dslist = [ifado.open_dataset(dtm=dtm) for dtm in pd.date_range("2010-01-01","2010-01-31")]
import pandas as pd
dslist = [ifado.open_dataset(dtm=dtm) for dtm in pd.date_range("2010-01-01","2010-01-31")]
xr.concat(dslist, dim="time")
import xarray as xr
xr.concat(dslist, dim="time")
dsm = _
dsm = dsm.mean(dim="time", skipna=True)
dsm
dsm.PP.max()
pprod = regions = kd_tree.resample_nearest(data=dsm.PP.data, **kw)
    kw = dict(source_geo_def=ifado.setup_grid(),
              target_geo_def=oc_cci.setup_grid(),
              radius_of_influence=50000)
pprod = regions = kd_tree.resample_nearest(data=dsm.PP.data, **kw)
from algorithms.export_production import dunne_2005
ds = model_to_field.open_dataset()
ep_dunne = dunne_2005(pprod=pprod*1000, sst=ds.sst.data, z_eup=ds.Zeu.data, chl=ds.chlor_a.data)
pcolormesh(squeeze(exp(ep_rf.export_production[0,:,:].data) - ep_dunne), cmap=cm.RdBu_r)
from remotefig import rfig
%pylab
ep_rf = xr.open_dataset("ncfiles/ep_no_regions_month_2010_zeu_depth.nc")
pcolormesh(squeeze(exp(ep_rf.export_production[0,:,:].data) - ep_dunne), cmap=cm.RdBu_r)
colorbar()
rfig()
clim(-1000,1000)
rfig()
clim(-100,100)
rfig()
history
history
"""
