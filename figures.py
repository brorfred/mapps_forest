from pathlib import Path

import numpy as np
import pylab as pl
from matplotlib import cm
import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.inspection import permutation_importance as sklearn_perm_imp

import projmap
from ocean_forest.config import settings


def scatter(model, ax=None, clf=True, x1=None, x2=None, dunne=False):
    if clf:
        pl.clf()
    ax = pl.gca() if ax is None else ax
    settings.setenv(getattr(model, 'env', 'default'))
    env = getattr(model, 'env', '')

    r2train = model.score(model.X_train, model.y_train)
    r2test  = model.score(model.X_test,  model.y_test)
    trainstr = f"Train (R$^2$={r2train:.2})"
    teststr  = f"Test (R$^2$={r2test:.2})"

    x1 = settings.x1 if x1 is None else x1
    x2 = settings.x2 if x2 is None else x2

    filename = f"{settings.fig_file_pref}_{env}_scatter.pdf"
    title   = "Random Forest Regression"
    if "depth" in model.X_train:
        title += " with 'depth'"
        filename = filename.replace(".pdf", "_depth.pdf")
    ax.set_title(title)
    ax.scatter(np.exp(model.y_train), np.exp(model.predict(model.X_train)), 
               5, label=trainstr, alpha=0.7, linewidths=0)
    ax.scatter(np.exp(model.y_test), np.exp(model.predict(model.X_test)),
               5, label=teststr, alpha=0.7, linewidths=0)
    pl.setp(ax, xscale="log", yscale="log", xlim=(x1,x2), ylim=(x1,x2))
    ax.plot([x1,x2], [x1,x2], "k", lw=1, alpha=0.2)
    if dunne:
        dfd = read_ep.dunne2005()  
        dfd["ep_pred"] = dunne_2005(pprod=dfd.pprod.values, sst=dfd.sst.values, 
                                   z_eup=dfd.Zeu.values, chl=dfd.chl.values)
        dfd = dfd[["ep_obs", "ep_pred"]].dropna()
        regr = linregress(dfd.ep_obs, dfd.ep_pred)
        dunnestr  = f"Dunne (R$^2$={regr.rvalue**2:.2})"

        pl.scatter(dfd.ep_obs, dfd.ep_pred, 30, label=dunnestr, 
                   c="tab:green", edgecolor="w", linewidths=0.4) 
        filename = filename.replace(".pdf", "_dunne.pdf")

    pl.xlabel(settings.scatter_xlabel)
    pl.ylabel(settings.scatter_ylabel)
    pl.legend()
    pl.savefig(f"figs/{filename}")

def decision_tree(model, tree=5):
    settings.setenv(getattr(model, 'env', 'default'))
    filename = f"{settings.fig_file_pref}_decision_tree_{tree:04}.dot"
    if "depth" in model.X_train:
        filename = filename.replace(".dot", "_depth.dot")
    estimator = model.estimators_[tree]
    export_graphviz(estimator, out_file="figs/"+filename,
        feature_names = model.X_train.keys(), class_names = model.y_train.name,
        rounded = True, proportion = False, precision = 2, filled = True)

def feature_importance(model=None):
    model = rf_regression() if model is None else model
    importance = model.feature_importances_

    pl.clf()
    ax = pl.gca()
    xlist = [x for x in range(len(importance))]
    pl.barh(xlist, importance)
    ax.set_yticks(xlist)
    ax.set_yticklabels(model.X_train.keys())

def permutation_importance(model):
    """Calculate permutation importances
    
    https://scikit-learn.org/stable/modules/permutation_importance.html
    """
    settings.setenv(getattr(model, 'env', 'default'))
    env = getattr(model, 'env', '')
    scoring = 'explained_variance'
    pimp = sklearn_perm_imp(model, model.X_test, model.y_test, 
                            n_repeats=30, random_state=0, scoring=scoring)
    perm_sorted_idx = pimp.importances_mean.argsort()
    tree_importance_sorted_idx = np.argsort(model.feature_importances_)
    tree_indices = np.arange(0, len(model.feature_importances_)) + 0.5
    pl.clf()
    pl.boxplot(
        pimp.importances[perm_sorted_idx].T,
        vert=False,
        labels=model.X_test.keys()[perm_sorted_idx],
    )
    pl.title("Permutation Feature Importance")
    pl.xlabel("Explained variance regression score")
    pl.xlim(0,1)
    #fig.tight_layout()

    filename = f"{settings.fig_file_pref}_{env}_permutation_feature_importances.pdf"
    if "depth" in model.X_train:
        filename = filename.replace(".pdf", "_depth.pdf")
    pl.savefig(f"figs/{filename}")

def depths(model):
    zvec = np.arange(0,500,5)

    X = pd.concat([model.X_test, model.X_train]).copy()
    y = np.exp(pd.concat([model.y_test, model.y_train]).copy())

    depth_bins = (X.depth/25).astype(int)*25
    real_depth = X["depth"]

    epavg = []
    epstd = []
    for depth in zvec:
        X.depth = depth
        yvec = np.exp(model.predict(X))
        epavg.append(np.median(yvec))
        epstd.append(np.std(yvec))
    epavg = np.array(epavg)
    epstd = np.array(epstd)
    X["depth_bins"] = depth_bins
    X["real_depth"] = real_depth
    X["EP_obs"] = y

    gr = X.groupby("depth_bins").median()
    size = X.groupby("depth_bins").size()

    pl.clf()
    pl.plot(epavg,       -zvec, "-", c="tab:blue", label="median predicted EP")
    pl.fill_betweenx(-zvec, epavg-epstd, epavg+epstd, 
                     color="tab:blue", alpha=0.1)
    pl.plot(epavg+epstd, -zvec, c="tab:blue", lw=0.5, alpha=0.5)
    pl.plot(epavg-epstd, -zvec, c="tab:blue", lw=0.5, alpha=0.5)
    pl.title(f"Mean Export production (n={len(X)})")
    pl.ylabel("Depth (m)")
    pl.xlabel("Export Production (mg C m$^{-2}$ d$^{-1}$)")
    pl.ylim(-500,0)
    pl.xlim(0,150)
    pl.legend()
    pl.savefig("figs/random_forest/EP_depths_clean.pdf")

    avgZeu = -np.median(X.Zeu)
    pl.plot([0,150], [avgZeu,avgZeu], c="tab:red", alpha=0.3, 
            label="median Z$_{eu}$")
    pl.legend()
    pl.savefig("figs/random_forest/EP_depths_zeu.pdf")

    pl.scatter(X.EP_obs, -X.real_depth, 1, c="tab:green", alpha=.5, 
               label="observed EP")
    pl.legend()
    pl.savefig("figs/random_forest/EP_depths_obs.pdf")


def residual(model, ax=None, clf=True, x1=None, x2=None):
    if clf:
        pl.clf()
    ax = pl.gca() if ax is None else ax
    settings.setenv(getattr(model, 'env', 'default'))
    env = getattr(model, 'env', '')
    x1 = settings.x1 if x1 is None else x1
    x2 = settings.x2 if x2 is None else x2


    filename = f"{settings.fig_file_pref}_{env}_residual.pdf"
    title   = "Residuals after RF Regression"
    if "depth" in model.X_train:
        title += " with depth"
        filename = filename.replace(".pdf", "_depth.pdf")

    ax.set_title(title)
    pl.scatter(np.exp(model.predict(model.X_train)), 
               np.exp(model.y_train - model.predict(model.X_train)), 5, label="Train")
    pl.scatter(np.exp(model.predict(model.X_test)), 
               np.exp(model.y_test - model.predict(model.X_test)), 5, label="Test")
    pl.setp(ax, xscale="log", xlim=(x1,x2))
    pl.setp(ax, yscale="log", ylim=(0.001,1000))

    ax.plot([x1,x2], [1,1], "k", alpha=0.2)
    pl.ylabel("Residual (mg C m$^{-2}$ d$^{-1}$)")
    pl.xlabel(settings.scatter_ylabel)
    pl.legend()
    pl.savefig(f"figs/{filename}")

def all_evaluation_figs(model):
    scatter(model)
    permutation_importance(model)
    residual(model)

def global_map(da, cmap=cm.nipy_spectral, title=""):
    pl.close("all")
    pl.figure(1, (10,8))
    mp = projmap.Map("glob")
    mp.style["oceancolor"] = "0.8"
    mp.pcolor(da.lon, da.lat, np.squeeze(da.data), cmap=cmap, 
        colorbar=True, vmin=0, vmax=150, rasterized=True)
    mp._cb.set_label("mg C m$^{-2}$")
    mp.nice()
    pl.text(0.5, 0.75, title, fontsize=18, horizontalalignment="center",
        transform=pl.gcf().transFigure)

def all_monthly_maps(ds):

    datadir = f"figs/global_maps/{ds.env}"
    Path(datadir).mkdir(parents=True, exist_ok=True)
    for da in ds.export_production:
        print(str(da.time))
        datestr = str(da.time.data)[:7]
        da.data[da.data==0] = np.nan
        global_map(da, title=datestr)
        pl.savefig(f"{datadir}/EP_global_map_{ds.env}_{datestr}.png", 
                   dpi=600, bbox_inches="tight")