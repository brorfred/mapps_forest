
"""Match satellite data to insitu observations in a pandas dataframe"""

import numpy as np
import pandas as pd
import xarray as xr

from njord2 import ostia, oc_cci_day, modis, seawifs, match, tiho_psd
from njord2 import oc_cci, longhurst
from oceandata import mapps as mapps_data
from utils import daylength as calc_daylength

def match_longhurst(df):
    mt = match.Match(longhurst)
    dtmvec = np.array(["2010-01-01"]*len(df))
    df["longhurst"] = mt.sameday(df.lon, df.lat, dtmvec, "regions", nei=1)
    df["basin"] =     mt.sameday(df.lon, df.lat, dtmvec, "basins", nei=1)
    df["biome"] =     mt.sameday(df.lon, df.lat, dtmvec, "biomes", nei=1)

def match_par(df, timetype="day", dtmvec=None, days_behind=30):
    dtmvec = df.index if dtmvec is None else dtmvec
    mask = df.index>"2003-01-01"
    marr = np.full((len(df), days_behind+1) , np.nan)
    if np.sum(mask) > 0:
        mt = match.Match(modis, dskw={"timetype":timetype})
        marr[mask] = np.squeeze(mt.multiday(df.lon[mask], df.lat[mask], dtmvec[mask], "par", 
                                            days_behind=days_behind))
    mask = df.index<"2004-12-31"
    sarr = np.full((len(df), days_behind+1), np.nan)
    if sum(mask) > 0:
        mt = match.Match(seawifs, dskw={"timetype":timetype})
        sarr[mask] = np.squeeze(mt.multiday(df.lon[mask], df.lat[mask], dtmvec[mask], "par",
                                            days_behind=days_behind))
    return np.nanmean((marr,sarr),axis=0)


def match_monthly(df):
    df = df[df.index >= "1998-01-01"]
    df = df[df.index.notnull()]
    df = df[df["lon"].notnull()]
    df = df[df["lat"].notnull()]

    df["month"] = df.index.month
    dtmvec = df.index.normalize().snap("MS") 
    mt = match.Match(ostia, dskw={"timetype":"mo"})
    df["sst"] = mt.sameday(df.lon, df.lat, dtmvec, "sst")
    mt = match.Match(oc_cci)
    df["chl"]    = mt.sameday(df.lon, df.lat, dtmvec, data_var="chlor_a")
    df["kd_490"] = mt.sameday(df.lon, df.lat, dtmvec, data_var="kd_490")
    df["Zeu"] = 4.6/df["kd_490"]
    match_par(df, timetype="mo", dtmvec=dtmvec)
    match_longhurst(df)
    return df

def match_daily(df):
    """Match satellite data to dataframe"""
    mt = match.Match(ostia)
    df = df[df.index >= "1998-01-01"]
    df["month"] = df.index.month
    df["sst"] = mt.sameday(df.lon, df.lat, df.index, "sst")
    mt = match.Match(oc_cci_day)
    df["chl"]    = mt.sameday(df.lon, df.lat, df.index, data_var="chlor_a")
    df["kd_490"] = mt.sameday(df.lon, df.lat, df.index, data_var="kd_490")
    df["Zeu"] = 4.6/df["kd_490"]
    match_par(df)
    match_longhurst(df)
    return df

def match_upstream(days_behind=30):
    """Match satellite data to dataframe"""
    df = mapps_data.load_pml()
    df = df[df.index >= (pd.to_datetime("1998-01-01") +  pd.Timedelta(f"{days_behind} days"))]
    df = df[~df[["lat","lon"]].isna().all(axis=1)]
    df["month"] = df.index.month
    #df = df.iloc[:10,:]
    df["daylength"] = calc_daylength(df.index.dayofyear, df.lat)

    mt = match.Match(ostia)
    dtmarr = mt.dtm_array(df.index, days_behind=days_behind)
    dlen = []
    for dtm in dtmarr.T: 
        dlen.append(calc_daylength(pd.to_datetime(dtm).dayofyear, df.lat))
    dlen = np.array(dlen).T
    ds["daylength"] = (("sample", "days_from_obs"), dlen)
    sst = mt.multiday(df.lon, df.lat, df.index, data_var="sst", days_behind=days_behind)
    mt = match.Match(oc_cci_day)
    chl = mt.multiday(df.lon, df.lat, df.index, data_var="chlor_a", days_behind=days_behind)
    kd_490 = mt.multiday(df.lon, df.lat, df.index, data_var="kd_490", days_behind=days_behind)
    Zeu = 4.6/kd_490
    par = match_par(df,days_behind=days_behind)
    match_longhurst(df)
    E_surf = par/dlen/3600*1e6
    Ez = E_surf * np.exp(kd_490*df["depth"].values[:,None])
    df["PB"] = df["PBmax"]*np.tanh(df["alpha"]*Ez[:,-1]/df["PBmax"])


    ds = xr.Dataset(data_vars={"sst":(("sample", "days_from_obs"), sst),
                               "chl":(("sample", "days_from_obs"), chl),
                               "kd_490":(("sample", "days_from_obs"), kd_490),
                               "par":(("sample", "days_from_obs"), par),
                               "Zeu":(("sample", "days_from_obs"), Zeu),
                               "E_surf":(("sample", "days_from_obs"), E_surf),
                               "Ez":(("sample", "days_from_obs"), Ez),
                               },
                    coords={"date":(("sample", "days_from_obs"), dtmarr),
                            "days_from_obs":range(-days_behind,1,1)}
                    )
    for key in ['ID', 'lat', 'lon', 'depth', 'temp', 'NO3', 'Si4', 'PO4', 
                'alpha', 'PBmax', 'Ek', 'month',
                'longhurst', 'basin', 'biome', 'PB']:
        ds[key] = (("sample",), df[key])
    return ds

def mean_upstream(days_behind=0, filename="mapps_upstream_sat.nc"):
    dfdict = {}
    ds = xr.open_dataset(filename)
    for key in ds.data_vars:
        if ds[key].ndim == 2:
            dfdict[key] = np.nanmean(ds[key][:,-1-days_behind:], axis=1)
        elif ds[key].dims == ("sample",):
            dfdict[key] = ds[key].values
    dfdict["date"] = ds.date[:,-1].values
    return pd.DataFrame(dfdict)

def count_upstream(days_behind=0, filename="mapps_upstream_sat.nc"):
    dfdict = {}
    ds = xr.open_dataset(filename)
    for key in ds.data_vars:
        if ds[key].ndim == 2:
            dfdict[key] = np.nansum(ds[key][:,-1-days_behind:]*0+1, axis=1)
    return pd.DataFrame(dfdict)

def std_upstream(days_behind=0, filename="mapps_upstream_sat.nc"):
    dfdict = {}
    ds = xr.open_dataset(filename)
    for key in ds.data_vars:
        if ds[key].ndim == 2:
            dfdict[key] = np.nanstd(ds[key][:,-1-days_behind:], axis=1)
    return pd.DataFrame(dfdict)

def all_upstream_files(daylist = range(0,30)):
    for days in daylist:
        df = mean_upstream(days_behind=days)
        df.to_csv(f"upstream_files/mapps_{days:02}_days_mean.csv")
        df = std_upstream(days_behind=days)
        df.to_csv(f"upstream_files/mapps_{days:02}_days_std.csv")
        df = count_upstream(days_behind=days)
        df.to_csv(f"upstream_files/mapps_{days:02}_days_count.csv")