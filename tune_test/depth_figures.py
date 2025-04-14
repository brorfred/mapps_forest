"""
refs
---
https://zillow.github.io/quantile-forest/user_guide.html
"""

import pylab as pl
import numpy as np
from quantile_forest import RandomForestQuantileRegressor

from ocean_forest.config import settings
from depth_profiles import expand_row_with_all_depths

def quantile_forest(model, quantiles=[0.05, 0.33, 0.5, 0.67, 0.95]):
    settings.setenv(model.env)
    model.qmodel = RandomForestQuantileRegressor(**settings["rf_params"])
    model.qmodel.fit(model.X_train, model.y_train)
    #model.q_train = qmodel.predict(model.X_train, quantiles=quantiles)
    #model.q_test = qmodel.predict(model.X_test, quantiles=quantiles)
    

def one_obs_scatter_quantile(model, pos=102, line=True):
    if not hasattr(model, "qmodel"):
        quantile_forest(model)
    qmodel = model.qmodel

    #41 37 12 70 83 84
    Xdf = model.X_test.iloc[[pos]]
    ydf = model.y_test[pos]
    linestr = ""

    pl.clf()
    depth = Xdf.depth
    depthstr = str(int(Xdf.depth.item()))
    pl.scatter(np.exp(ydf), depth, c="tab:orange", label="observation")
    pl.scatter(np.exp(qmodel.predict(Xdf)), depth, c="tab:blue", label="prediction")
    zeu = Xdf.Zeu.values.item()
    if zeu > 100:
        pl.plot([-5,150], [zeu, zeu], lw=1, c="tab:green", label="Z$_{eu}^{1\%}$")
    if line:
        df = expand_row_with_all_depths(Xdf)
        qarr = np.exp(qmodel.predict(df, quantiles=[0.05, 0.33, 0.5, 0.67, 0.95]))
        pl.fill_betweenx(df.depth, qarr[:,0], qarr[:,4], alpha=0.25, color="tab:blue")
        pl.fill_betweenx(df.depth, qarr[:,1], qarr[:,3], alpha=0.25, color="tab:blue")
        pl.plot(qarr[:,2], df.depth,  c="tab:blue")
        depthstr = "0, 5, ..., 500"
        linestr = "line"

    facstr1 = (f"Chl={np.exp(Xdf.chl.item()):.2f}, " + 
               f"PAR={Xdf.sat_par.item():.2f}, " + 
               f"Zeu={Xdf.Zeu.item():.2f}, " + 
               f"kd$_{{490}}$={Xdf.kd_490.item():.2f}, ")
    facstr2 = (f"SST={Xdf.sst.item():.2f}, " + 
               f"month={Xdf.month.item()}, " +
               f"basin={int(Xdf.basin.item())}, " + 
               f"depth={depthstr}")

    pl.text(50,460, facstr1, size="small")    
    pl.text(50,480, facstr2, size="small")
    pl.xlim(-5, 150)
    pl.ylim(500, 100)
    pl.legend(loc="center right")
    pl.xlabel("POC flux (mg C d$^{-1}$)")
    pl.ylabel("depth (m)")
    #pl.savefig(f"figs/example_profile/scatter_{linestr}_{pos:03}.pdf")

def one_obs_scatter(model, pos=102, line=True):
    #41 37 12 70 83 84
    Xdf = model.X_test.iloc[[pos]]
    ydf = model.y_test[pos]
    linestr = ""

    pl.clf()
    depth = Xdf.depth
    depthstr = str(int(Xdf.depth.item()))
    pl.scatter(np.exp(ydf), depth, c="tab:orange", label="observation")
    pl.scatter(np.exp(model.predict(Xdf)), depth, c="tab:blue", label="prediction")
    zeu = Xdf.Zeu.values.item()
    if zeu > 100:
        pl.plot([-5,150], [zeu, zeu], lw=1, c="tab:green", label="Z$_{eu}^{1\%}$")
    if line:
        df = expand_row_with_all_depths(Xdf)
        qarr = np.exp(model.predict(df))
        pl.plot(qarr, df.depth,  c="tab:blue")
        depthstr = "0, 5, ..., 500"
        linestr = "line"

    facstr1 = (f"Chl={np.exp(Xdf.chl.item()):.2f}, " + 
               f"PAR={Xdf.sat_par.item():.2f}, " + 
               f"Zeu={Xdf.Zeu.item():.2f}, " + 
               f"kd$_{{490}}$={Xdf.kd_490.item():.2f}, ")
    facstr2 = (f"SST={Xdf.sst.item():.2f}, " + 
               f"month={Xdf.month.item()}, " +
               f"basin={int(Xdf.basin.item())}, " + 
               f"depth={depthstr}")

    pl.text(50,460, facstr1, size="small")    
    pl.text(50,480, facstr2, size="small")
    pl.xlim(-5, 150)
    pl.ylim(500, 100)
    pl.legend(loc="center right")
    pl.xlabel("POC flux (mg C d$^{-1}$)")
    pl.ylabel("depth (m)")
    #pl.savefig(f"figs/example_profile/scatter_{linestr}_{pos:03}.pdf")




def vertical_curve(model):
    df = load()
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

