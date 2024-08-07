import numpy as np
from numba import jit

@jit(nopython=True)
def bias(evaluation, simulation):
    """
    Bias objective function
    """
    return np.nansum(evaluation - simulation) / len(evaluation)

@jit(nopython=True)
def mielke_skill_score(evaluation, simulation):
    """ Mielke index 
    if pearson coefficient (r) is zero or positive use kappa=0
    otherwise see Duveiller et al. 2015
    """
    x = evaluation
    y = simulation
    mx = np.mean(x)
    my = np.mean(y)
    xm, ym = x-mx, y-my

    diff= (evaluation - simulation) ** 2 
    d1= np.sum(diff)
    d2= np.var(evaluation)+np.var(simulation)+ (np.mean(evaluation)-np.mean(simulation))**2
    
    if correlation_coefficient_loss(evaluation, simulation) < 0:
        kappa = np.abs( np.sum(xm*ym)) * 2
        mss= 1-(  ( d1* (1/len(evaluation))  ) / (d2 +kappa))
    else:
        mss= 1-(  ( d1* (1/len(evaluation))  ) / d2 )

    return mss

@jit(nopython=True)
def correlation_coefficient_loss(evaluation, simulation):
    x = evaluation
    y = simulation
    mx = np.mean(x)
    my = np.mean(y)
    xm, ym = x-mx, y-my
    r_num = np.sum(xm*ym)
    r_den = np.sqrt(np.sum(np.square(xm)) * np.sum(np.square(ym)))
    r = r_num / r_den
    r = np.maximum(np.minimum(r, 1.0), -1.0)

    return 1- np.square(r)

@jit(nopython=True)
def nashsutcliffe(evaluation, simulation):
    """
    Nash-Sutcliffe objective function
    """
    return 1 - np.sum((evaluation - simulation) ** 2) / np.sum((evaluation - np.mean(evaluation)) ** 2)

@jit(nopython=True)
def lognashsutcliffe(evaluation, simulation):
    """
    Log Nash-Sutcliffe objective function
    """
    return 1 - sum((np.log(simulation) - np.log(evaluation)) ** 2) / sum((np.log(evaluation) - np.mean(np.log(evaluation))) ** 2)

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
        numerator = 2 * np.sum(x_rank * y_rank) - n * (n - 1)
        denominator = n * (n - 1) * (n + 1)
        return numerator / denominator
    
@jit(nopython=True)
def agreementindex(evaluation, simulation):
    """
    Agreement Index
    """
    return 1 - (np.sum((evaluation - simulation) ** 2)) / (np.sum((np.abs(simulation - np.mean(evaluation)) + np.abs(evaluation - np.mean(evaluation)))** 2))

@jit(nopython=True)
def kge(evaluation, simulation):
    mu_s = np.mean(simulation)
    mu_o = np.mean(evaluation)
    if mu_o == 0:
        mu_o = 1e-10  # Pequeno valor para evitar divisão por zero
    std_s = np.std(simulation)
    std_o = np.std(evaluation)
    if std_o == 0:
        std_o = 1e-10  # Pequeno valor para evitar divisão por zero
    
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
    
    sim_mean = np.mean(simulation)
    eval_mean = np.mean(evaluation)

    if eval_mean == 0:
        eval_mean = 1e-10  # Pequeno valor para evitar divisão por zero
    if sim_mean == 0:
        sim_mean = 1e-10  # Pequeno valor para evitar divisão por zero
    
    fdc_sim = np.sort(simulation / (sim_mean * len(simulation)))
    fdc_obs = np.sort(evaluation / (eval_mean * len(evaluation)))
    
    alpha = 1 - 0.5 * np.mean(np.abs(fdc_sim - fdc_obs))
    beta = sim_mean / eval_mean
    
    kge = 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    
    return kge

@jit(nopython=True)
def log_p(evaluation, simulation):
    """
    Logarithmic Probability Distribution
    """
    scale = np.mean(evaluation) / 10
    if scale < 0.01:
        scale = 0.01
    y = (np.array(evaluation) - np.array(simulation)) / scale
    normpdf = -(y**2) / 2 - np.log(np.sqrt(2 * np.pi))
    return np.mean(normpdf)

@jit(nopython=True)
def covariance(evaluation, simulation):
    """
    Covariance objective function
    """
    obs_mean = np.mean(evaluation)
    sim_mean = np.mean(simulation)
    covariance = np.mean((evaluation - obs_mean) * (simulation - sim_mean))
    return covariance

@jit(nopython=True)
def pbias(evaluation, simulation):
    """
    Percent Bias
    """
    return 100 * np.sum(evaluation - simulation) / np.sum(evaluation)

@jit(nopython=True)
def mse(evaluation, simulation):
    """
    Mean Squared Error
    """
    return np.mean((evaluation - simulation) ** 2)

@jit(nopython=True)
def rmse(evaluation, simulation):
    """
    Root Mean Squared Error
    """
    return np.sqrt(np.mean((evaluation - simulation) ** 2))

@jit(nopython=True)
def mae(evaluation, simulation):
    """
    Mean Absolute Error
    """
    return np.mean(np.abs(evaluation - simulation))

@jit(nopython=True)
def rrmse(evaluation, simulation):
    """
    Relative RMSE
    """
    return rmse(evaluation, simulation) / np.mean(evaluation)

@jit(nopython=True)
def rsr(evaluation, simulation):
    """
    RMSE-observations standard deviation ratio
    """
    return rmse(evaluation, simulation) / np.std(evaluation)

@jit(nopython=True)
def decomposed_mse(evaluation, simulation):
    """
    Decomposed MSE
    """
    e_std = np.std(evaluation)
    s_std = np.std(simulation)
    bias_squared = bias(evaluation, simulation) ** 2
    sdsd = (e_std - s_std) ** 2
    lcs = 2 * e_std * s_std * (1 - np.corrcoef(evaluation, simulation)[0, 1])
    decomposed_mse = bias_squared + sdsd + lcs

    return decomposed_mse

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
    True,
]

# Now we must provide a list with all the functions that will be used in the calibration process
opt = [
    mielke_skill_score,
    nashsutcliffe,
    lognashsutcliffe,
    pearson,
    spearman,
    agreementindex,
    kge,
    npkge,
    log_p,
    bias,
    pbias,
    mse,
    rmse,
    mae,
    rrmse,
    rsr,
    covariance,
    decomposed_mse,
]