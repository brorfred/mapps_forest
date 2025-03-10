

import shap 

import matplotlib.pyplot as plt

def calc_shap(model):
    X = pd.concat((model.X_train,model.X_test))
    explainer = shap.TreeExplainer(model=model, data=X)
    shap_values = explainer.shap_values(X)
    shap_values.X = X
    return shap_values
    
def bar(shap_values):
    plt.clf()
    shap.plots.bar(shap_values)
    plt.savefig("figs/shap/bar.pdf", dpi=900, bbox_inches="tight")

def bees(shap_values):
    plt.clf()
    shap.plots.beeswarm(shap_values)
    plt.savefig("figs/shap/beeswarm.pdf", dpi=900, bbox_inches="tight")

def violin(shap_values):
    X = shap_values.X
    feat_names = list(X.columns)
    plt.clf()
    shap.plots.violin(shap_values, features=X, feature_names=feat_names, plot_type="layered_violin")
    plt.savefig("figs/shap/violin.pdf", dpi=900, bbox_inches="tight")

def dependence(shap_values):
    X = shap_values.X
    plt.clf()
    shap.dependence_plot("kd_490", shap_values.values, X)
    plt.savefig("figs/shap/shap_dependence.pdf", dpi=900, bbox_inches="tight")


"""
shap.dependence_plot("kd_490", shap_values, X)
savefig("temp.png", dpi=900)
setp(gca(), xscale="log")
savefig("temp.png", dpi=900)
setp(gca(), xscale="lin")
setp(gca(), xscale="linear")
savefig("temp.png", dpi=900)
shap.dependence_plot("sst", shap_values, X)
savefig("temp.png", dpi=900)
shap.dependence_plot("chl", shap_values, X)
savefig("temp.png", dpi=900)
savefig?
savefig("temp.png", dpi=900, bbox_inches="tight")
history"""