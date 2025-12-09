import numpy as np
from scipy.optimize import curve_fit

def log_gompertz_makeham(x, a, b, c):
    """
    Logarithm of the Gompertz-Makeham mortality model.
    M(x) = a * exp(b * x) + c
    log(M(x)) = log(a * exp(b * x) + c)
    """
    return np.log(a * np.exp(b * x) + c)

def log_gompertz(x, a, b):
    """
    Logarithm of the Gompertz mortality model.
    M(x) = a * exp(b * x)
    log(M(x)) = log(a) + b * x
    """
    return np.log(a) + b * x

def fit_gompertz(mortality_rates, use_makeham=False, fit_region=[25, 100]):
    """
    Fit Gompertz or Gompertz-Makeham model to mortality rates.
    
    Parameters
    ----------
    mortality_rates : pd.Series or np.array
        Mortality rates indexed by age (or array where index is age)
    use_makeham : bool
        Whether to include the Makeham term (constant c)
    fit_region : list
        [start_age, end_age] to use for fitting
        
    Returns
    -------
    tuple
        (x_full, y_pred, params, cov)
        x_full: array of all ages
        y_pred: predicted mortality rates for all ages
        params: fitted parameters (a, b) or (a, b, c)
        cov: covariance matrix
    """
    # Convert to numpy array if pandas Series
    if hasattr(mortality_rates, 'values'):
        y_values = mortality_rates.values
        x_full = mortality_rates.index.values.astype(int)
    else:
        y_values = np.array(mortality_rates)
        x_full = np.arange(len(y_values))
        
    # Handle fit region
    start_idx = fit_region[0]
    end_idx = fit_region[1]
    
    # Ensure indices are within bounds
    if start_idx < 0: start_idx = 0
    if end_idx > len(y_values): end_idx = len(y_values)
    
    x_trimmed = x_full[start_idx:end_idx]
    y_trimmed = y_values[start_idx:end_idx]
    
    # Log transform for fitting
    # Handle zeros or negative values if any (though mortality shouldn't be <= 0)
    mask = y_trimmed > 0
    x_fit = x_trimmed[mask]
    y_fit = y_trimmed[mask]
    y_log = np.log(y_fit)
    
    if len(x_fit) < 3:
        raise ValueError("Not enough data points for fitting in the specified region.")

    if use_makeham:
        # Initial guess: a=0.0001, b=0.08, c=0.0001
        p0 = (0.0001, 0.08, 0.0001)
        # Bounds: a>0, b>0, c>=0
        bounds = ([1e-12, 1e-12, 0], [1, 1, 1])
        
        try:
            params, cov = curve_fit(log_gompertz_makeham, x_fit, y_log, p0=p0, bounds=bounds, maxfev=10000)
            y_pred = np.exp(log_gompertz_makeham(x_full, *params))
        except RuntimeError:
            # Fallback if fit fails
            params = (0, 0, 0)
            cov = None
            y_pred = np.zeros_like(x_full)
    else:
        # Initial guess: a=0.0001, b=0.08
        p0 = (0.0001, 0.08)
        # Bounds: a>0, b>0
        bounds = ([1e-12, 1e-12], [1, 1])
        
        try:
            params, cov = curve_fit(log_gompertz, x_fit, y_log, p0=p0, bounds=bounds, maxfev=10000)
            y_pred = np.exp(log_gompertz(x_full, *params))
        except RuntimeError:
            params = (0, 0)
            cov = None
            y_pred = np.zeros_like(x_full)
    
    return x_full, y_pred, params, cov

def get_gompertz_equation(a, b, c=None):
    """
    Format the Gompertz equation string.
    """
    if c is not None:
        return f"M(t) = {a:.2e} * exp({b:.4f} * t) + {c:.2e}"
    else:
        return f"M(t) = {a:.2e} * exp({b:.4f} * t)"

