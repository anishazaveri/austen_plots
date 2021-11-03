import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import mean_squared_error


def calc_delta(alpha: float, g: pd.Series, bias: float):
    expectation_term = np.mean((1 / g) + (1 / (1 - g)))
    delta = bias * ((1 / alpha) - 1) / expectation_term
    return delta


def calc_Rsq(row: pd.Series, g: pd.Series, Q: pd.Series, t: pd.Series, y: pd.Series):
    alpha = row['alpha']
    delta = row['delta']
    alpha_shape = g * ((1 / alpha) - 1)
    beta_shape = (1 - g) * ((1 / alpha) - 1)
    Rsq_num = delta ** 2 * np.mean(sp.special.polygamma(1, alpha_shape +
                                                        t) + sp.special.polygamma(1, beta_shape + (1 - t)))
    Rsq_den = mean_squared_error(y, Q)
    return Rsq_num / Rsq_den


def calc_Rsqhat(y: pd.Series, Qhat: pd.Series, Q: pd.Series):
    Rsqhat = (mean_squared_error(y, Qhat) - mean_squared_error(y, Q)) / \
             (mean_squared_error(y, Qhat))
    if Rsqhat < 0:
        Rsqhat = 0
    return Rsqhat


def calc_ahat(ghat: pd.Series, g: pd.Series):
    ahat = 1 - (np.mean(g * (1 - g)) / np.mean(ghat * (1 - ghat)))
    if ahat < 0:
        ahat = 0
    return ahat


def calc_ci(row: pd.Series, alpha_ci: float):
    boot_values = row[[col for col in list(row.index) if '_' in col]].values
    main_value = row[[col for col in list(
        row.index) if '_' not in col]].values[0]
    boot_values_d = boot_values - main_value
    lower_quant = np.quantile(boot_values_d, (1 - alpha_ci) / 2)
    upper_quant = np.quantile(boot_values_d, alpha_ci + ((1 - alpha_ci) / 2))
    lower_lim = main_value - upper_quant
    upper_lim = main_value - lower_quant
    if lower_lim < 0:
        lower_lim = 0
    if upper_lim > 1:
        upper_lim = 1
    return main_value, lower_lim, upper_lim
