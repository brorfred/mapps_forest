# mapps_forest

mapps_forest is an implementation of the ocean_forest Python package to perform Random Forest regressions based on in-situ observations of ocean properties. This case is set up for the MAPPS data set.

## Installation

The easiest approach to install dependencies is to use conda. Just create a virtual envirnment from the included environment.yml file:

```bash
conda env create -f environment.yml 
```

## Usage

```python

import ocean_forest

# Load data
df = ocean_forest.load(env="mapps-alpha-daily")
# Fit a Random forest model to data
ocean_forest.regress(env="mapps-alpha-daily", depths=True, min_samples_leaf=8)
#Available hyperparameters can be found at
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

# Generate figures:
figures.all_evaluation_figs(model)

# Hyper parameters and other presets are set in the 'settings.toml' file 
# Existing default models are saved in the subfolder rf_models

```

# References
https://github.com/zillow/quantile-forest
https://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf

