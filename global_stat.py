
import calendar

import pylab as pl
import xarray as xr
import numpy as np
import pandas as pd

def one_year_lines(year=2010):
    area = xr.open_dataset("ncfiles/nasa_4km_area.nc")["area"].data

    pl.clf()
    #ds = xr.open_dataset(f"ncfiles/ep_no_regions_month_{year}.nc")
    ds = xr.open_dataset(f"ncfiles/EP_below_zeu_no_regions_depths_depth-100_{year}.nc")
    ep100 = [365 * np.nansum(ds.export_production[mn,:,:] * area) / 1e18 for mn in range(12)]
    pl.plot(calendar.month_abbr[1:],ep100, label="z=100m")

    ds = xr.open_dataset(f"ncfiles/EP_below_zeu_no_regions_depths_depth-Zeu_{year}.nc")
    epzeu = [365 * np.nansum(ds.export_production[mn,:,:] * area) / 1e18 for mn in range(12)]
    pl.plot(calendar.month_abbr[1:],epzeu, label="z=Zeu")

    ds = xr.open_dataset(f"ncfiles/EP_below_zeu_no_regions_depths_depth-Zeu01_{2010}.nc")
    epz01 = [365 * np.nansum(ds.export_production[mn,:,:] * area) / 1e18 for mn in range(12)]
    pl.plot(calendar.month_abbr[1:],epz01, label="z=Zeu$_{0.1\%}$")
    pl.ylim(0,10)
    pl.ylabel("POC flux (Pg C y$^{-1}$)")
    pl.legend()
    pl.title(f"Global Export Production {year}")

def all_year_lines():
    area = xr.open_dataset("infiles/nasa_4km_area.nc")["area"].data
    tvec = pd.date_range("2003-01-01","2020-12-31",freq="MS")
    fnprefix = "infiles/EP_below_zeu_no_regions_depths_depth"
    def calc_tvec(depth):
        eplist = []
        for year in range(2003, 2021):
            print(depth, year)
            ds = xr.open_dataset(f"{fnprefix}-{depth}_{year}.nc")
            epzeu = [365 * np.nansum(ds.export_production[mn,:,:] * area) / 1e18
                    for mn in range(12)]
            #pl.plot(calendar.month_abbr[1:],epzeu, label="z=Zeu")
            eplist.append(epzeu)
        return np.hstack(eplist)
    epzeu = calc_tvec("Zeu")
    epz01 = calc_tvec("Zeu01")
    ep100 = calc_tvec("100")
    return pd.DataFrame({"100":ep100, "Zeu":epzeu, "Z01":epz01}, index=tvec)
    #ds = xr.open_dataset(f"ncfiles/EP_below_zeu_no_regions_depths_depth-Zeu01_{2010}.nc")
    #epz01 = [365 * np.nansum(ds.export_prodution[mn,:,:] * area) / 1e18 for mn in range(12)]
    #pl.plot(calendar.month_abbr[1:],epz01, label="z=Zeu$_{0.1\%}$")
    pl.ylim(0,10)
    pl.ylabel("POC flux (Pg C y$^{-1}$)")
    pl.legend()
    pl.title(f"Global Export Production")

def plot_global_timeseries():

     tvec = pd.date_range("2003-01-01","2020-12-31",freq="MS")
