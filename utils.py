import numpy as np

import sklearn.metrics 

def daylength(dayOfYear, lat):
    """Computes the length of the day (the time between sunrise and
    sunset) given the day of the year and latitude of the location.
    Function uses the Brock model for the computations.
    For more information see, for example,
    Forsythe et al., "A model comparison for daylength as a
    function of latitude and day of year", Ecological Modelling,
    1995.
    Parameters
    ----------
    dayOfYear : int
        The day of the year. 1 corresponds to 1st of January
        and 365 to 31st December (on a non-leap year).
    lat : float
        Latitude of the location in degrees. Positive values
        for north and negative for south.
    Returns
    -------
    d : float
        Daylength in hours.
    """
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45*np.sin(np.deg2rad(360.0*(283.0+dayOfYear)/365.0))
    mask = -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))
    hourAngle = np.rad2deg(np.arccos(-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))))
    day_hours = 2.0*hourAngle/15.0
    day_hours[mask<=-1.0] = 24.0
    day_hours[mask>= 1.0] = 0
    return day_hours

def calc_r2(obs, est):
    r2mask = np.isfinite(obs) & np.isfinite(est)
    obs = obs[r2mask]
    est = est[r2mask]
    return 1 - np.sum(((obs - est)**2)) / np.sum(((obs-np.mean(obs))**2))


def metrics(obs, est, name="r2_score", **kw):
    mask = np.isfinite(obs) & np.isfinite(est)
    obs = obs[mask]
    est = est[mask]
    if name == "bias":
        return np.mean(est-obs)
    elif name == "smape":
        return 100/len(obs) * np.sum(2 * np.abs(est - obs) / (np.abs(obs) + np.abs(est)))
    func = getattr(sklearn.metrics, name)
    return  func(obs, est, **kw)
