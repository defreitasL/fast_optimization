import numpy as np
from numba import jit


@jit(nopython=True)
def bias(evaluation, simulation):
    """
    Bias objective function
    """
    return np.nansum(evaluation - simulation) / len(evaluation)

@jit(nopython=True)
def correlation_coefficient_loss(evaluation, simulation):
    x = evaluation
    y = simulation
    mx = np.nanmean(x)
    my = np.nanmean(y)
    xm, ym = x - mx, y - my
    r_num = np.nansum(xm * ym)
    r_den = np.sqrt(np.nansum(np.square(xm)) * np.nansum(np.square(ym)))
    if r_den == 0:
        r_den = 1e-10
    r = r_num / r_den
    r = np.maximum(np.minimum(r, 1.0), -1.0)
    return 1 - np.square(r)

@jit(nopython=True)
def mielke_skill_score(evaluation, simulation):
    """ Mielke index 
    if pearson coefficient (r) is zero or positive use kappa=0
    otherwise see Duveiller et al. 2015
    """
    x = evaluation
    y = simulation
    mx = np.nanmean(x)
    my = np.nanmean(y)
    xm, ym = x - mx, y - my

    diff = (evaluation - simulation) ** 2 
    d1 = np.nansum(diff)
    d2 = np.nanvar(evaluation) + np.nanvar(simulation) + (np.nanmean(evaluation) - np.nanmean(simulation)) ** 2
    
    if correlation_coefficient_loss(evaluation, simulation) < 0:
        kappa = np.abs(np.nansum(xm * ym)) * 2
        mss = 1 - (d1 / len(evaluation)) / (d2 + kappa)
    else:
        mss = 1 - (d1 / len(evaluation)) / d2

    return mss

@jit(nopython=True)
def nashsutcliffe(evaluation, simulation):
    """
    Nash-Sutcliffe objective function
    """
    return 1 - np.nansum((evaluation - simulation) ** 2) / np.nansum((evaluation - np.nanmean(evaluation)) ** 2)

@jit(nopython=True)
def lognashsutcliffe(evaluation, simulation):
    """
    Log Nash-Sutcliffe objective function
    """
    return 1 - np.nansum((np.log(simulation) - np.log(evaluation)) ** 2) / np.nansum((np.log(evaluation) - np.nanmean(np.log(evaluation))) ** 2)

@jit(nopython=True)
def pearson(evaluation, simulation):
    """
    Pearson objective function
    """
    return np.corrcoef(evaluation, simulation)[0, 1]

@jit(nopython=True)
def spearman(x, y):
    """
    Calculate Spearman's rank correlation coefficient.
    """
    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))
    n = len(x)
    if n == 0:
        return 0
    else:
        numerator = 2 * np.nansum(x_rank * y_rank) - n * (n - 1)
        denominator = n * (n - 1) * (n + 1)
        return numerator / denominator
    
@jit(nopython=True)
def agreementindex(evaluation, simulation):
    """
    Agreement Index
    """
    return 1 - (np.nansum((evaluation - simulation) ** 2)) / (np.nansum((np.abs(simulation - np.nanmean(evaluation)) + np.abs(evaluation - np.nanmean(evaluation))) ** 2))

@jit(nopython=True)
def kge(evaluation, simulation):
    mu_s = np.nanmean(simulation)
    mu_o = np.nanmean(evaluation)
    if mu_o == 0:
        mu_o = 1e-10
    std_s = np.nanstd(simulation)
    std_o = np.nanstd(evaluation)
    if std_s == 0:
        std_s = 1e-10
    
    r = np.corrcoef(simulation, evaluation)[0, 1]
    beta = mu_s / mu_o
    alpha = std_s / std_o
    
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge

@jit(nopython=True)
def npkge(evaluation, simulation):
    """
    Non parametric Kling-Gupta Efficiency

    Corresponding paper:
    Pool, Vis, and Seibert, 2018 Evaluating model performance: towards a non-parametric variant of the Kling-Gupta efficiency, Hydrological Sciences Journal.

    output:
        kge: Kling-Gupta Efficiency

    author: Nadine Maier and Tobias Houska
    optional_output:
        cc: correlation
        alpha: ratio of the standard deviation
        beta: ratio of the mean
    """
    cc = spearman(evaluation, simulation)
    
    sim_mean = np.nanmean(simulation)
    eval_mean = np.nanmean(evaluation)
    if eval_mean == 0:
        eval_mean = 1e-10
    if sim_mean == 0:
        sim_mean = 1e-10
    
    fdc_sim = np.sort(simulation / (sim_mean * len(simulation)))
    fdc_obs = np.sort(evaluation / (eval_mean * len(evaluation)))
    
    alpha = 1 - 0.5 * np.nanmean(np.abs(fdc_sim - fdc_obs))
    beta = sim_mean / eval_mean
    
    kge = 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    
    return kge

@jit(nopython=True)
def log_p(evaluation, simulation):
    """
    Logarithmic Probability Distribution
    """
    scale = np.nanmean(evaluation) / 10
    if scale < 0.01:
        scale = 0.01
    y = (evaluation - simulation) / scale
    normpdf = -(y ** 2) / 2 - np.log(np.sqrt(2 * np.pi))
    return np.nanmean(normpdf)

@jit(nopython=True)
def covariance(evaluation, simulation):
    """
    Covariance objective function
    """
    obs_mean = np.nanmean(evaluation)
    sim_mean = np.nanmean(simulation)
    covariance = np.nanmean((evaluation - obs_mean) * (simulation - sim_mean))
    return covariance

@jit(nopython=True)
def pbias(evaluation, simulation):
    """
    Percent Bias
    """
    return 100 * np.nansum(evaluation - simulation) / np.nansum(evaluation)

@jit(nopython=True)
def mse(evaluation, simulation):
    """
    Mean Squared Error
    """
    return np.nanmean((evaluation - simulation) ** 2)

@jit(nopython=True)
def rmse(evaluation, simulation):
    """
    Root Mean Squared Error
    """
    return np.sqrt(np.nanmean((evaluation - simulation) ** 2))

@jit(nopython=True)
def mae(evaluation, simulation):
    """
    Mean Absolute Error
    """
    return np.nanmean(np.abs(evaluation - simulation))

@jit(nopython=True)
def rrmse(evaluation, simulation):
    """
    Relative RMSE
    """
    return rmse(evaluation, simulation) / np.nanmean(evaluation)

@jit(nopython=True)
def rsr(evaluation, simulation):
    """
    RMSE-observations standard deviation ratio
    """
    return rmse(evaluation, simulation) / np.nanstd(evaluation)

@jit(nopython=True)
def decomposed_mse(evaluation, simulation):
    """
    Decomposed MSE
    """
    e_std = np.nanstd(evaluation)
    s_std = np.nanstd(simulation)
    bias_squared = bias(evaluation, simulation) ** 2
    sdsd = (e_std - s_std) ** 2
    lcs = 2 * e_std * s_std * (1 - np.corrcoef(evaluation, simulation)[0, 1])
    decomposed_mse = bias_squared + sdsd + lcs

    return decomposed_mse

# @jit
def backtot():
    metrics_name_list = [
        'mss',                      #Max Mielke Skill Score (MSS) ok
        'nashsutcliffe',            #Max Nash-Sutcliffe Efficiency (NSE) ok
        'lognashsutcliffe',         #Max log(NSE) ok
        'pearson',                  #Max Pearson Correlation ($\rho$) ok
        'spearman',                 #Max Spearman Correlation ($S_{rho}$) ok
        'agreementindex',           #Max Agreement Index (AI) ok
        'kge',                      #Max Kling-Gupta Efficiency (KGE) ok
        'npkge',                    #Max Non-parametric KGE (npKGE) ok
        'log_p',                    #Max Logarithmic Probability Distribution (LPD) ok
        'bias',                     #Min Bias (BIAS) ok
        'pbias',                    #Min Percent Bias (PBIAS) ok
        'mse',                      #Min Mean Squared Error (MSE) ok
        'rmse',                     #Min Root Mean Squared Error (RMSE) ok
        'mae',                      #Min Mean Absolute Error (MAE) ok
        'rrmse',                    #Min Relative RMSE (RRMSE) ok
        'rsr',                      #Min RMSE-observations standard deviation ratio (RSR) ok
        'covariance',               #Min Covariance ok
        'decomposed_mse',           #Min Decomposed MSE (DMSE) ok
    ]

    mask = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    return metrics_name_list, mask

# @jit(nopython=True)
def opt(index, evaluation, simulation):

    if index == 0:
        out = mielke_skill_score(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = -999
        if np.isnan(out):# or out < 0:
            return -999
        return out
    elif index == 1:
        out = nashsutcliffe(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = -999
        if np.isnan(out):# or out < 0:
            return -999
        return out
    elif index == 2:
        out = lognashsutcliffe(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = -999
        if np.isnan(out):# or out < 0:
            return -999
        return out
    elif index == 3:
        out = pearson(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = np.nan
        if np.isnan(out):# or out < 0:
            return 1e-6
        return out
    elif index == 4:
        out = spearman(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = np.nan
        if np.isnan(out):# or out < 0:
            return 1e-6
        return out
    elif index == 5:
        out = agreementindex(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = np.nan
        if np.isnan(out):# or out < 0:
            return 1e-6
        return out
    elif index == 6:
        out = kge(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = -999
        if np.isnan(out):# or out < 0:
            return -999
        return out
    elif index == 7:
        out = npkge(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = np.nan
        if np.isnan(out):# or out < 0:
            return 1e-6
        return out
    elif index == 8:
        out = log_p(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = np.nan
        if np.isnan(out):# or out < 0:
            return -999
        return out
    elif index == 9:
        out = bias(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = np.nan
        if np.isnan(out):
            return 1000
        return np.abs(out)
    elif index == 10:
        out = pbias(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = np.nan
        if np.isnan(out):
            return 100
        return out
    elif index == 11:
        out = mse(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = np.nan
        if np.isnan(out):
            return 1000
        return out
    elif index == 12:
        out = rmse(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = np.nan
        if np.isnan(out):
            return 1000
        return out
    elif index == 13:
        out = mae(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = np.nan
        if np.isnan(out):
            return 1000
        return np.abs(out)
    elif index == 14:
        out = rrmse(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = np.nan
        if np.isnan(out):
            return 1000
        return np.abs(out)
    elif index == 15:
        out = rsr(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = np.nan
        if np.isnan(out):
            return 1000
        return np.abs(out)
    elif index == 16:
        out = covariance(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = np.nan
        if np.isnan(out):
            return 1000
        return np.abs(out)
    elif index == 17:
        out = decomposed_mse(evaluation, simulation)
        if np.abs(out) > 1e6:
            out = np.nan
        if np.isnan(out):
            return 1000
        return out
    else:
        Warning('Invalid index')
