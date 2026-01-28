# functions.py
import numpy as np
import pandas as pd
from scipy.stats import norm

# -------------------------
# CAMB wrapper
# -------------------------
def camb_calc(cosmo_params,
              AccuracyBoost=1, lAccuracyBoost=1, lSampleBoost=1,
              input_data=None):
    import camb
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmo_params[0],
                       ombh2=cosmo_params[1],
                       omch2=cosmo_params[2],
                       mnu=cosmo_params[3],
                       num_massive_neutrinos=1,
                       tau=0.054)
    
    pars.InitPower.set_params(As=cosmo_params[4], ns=cosmo_params[5])
    pars.set_accuracy(AccuracyBoost=AccuracyBoost,
                      lAccuracyBoost=lAccuracyBoost,
                      lSampleBoost=lSampleBoost)
    pars.set_matter_power(redshifts=[0.0], kmax=20.0)
    results = camb.get_results(pars)
    k, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10.0, npoints=200)
    return pd.DataFrame({"pk": pk[0]})

# -------------------------
# CLASS wrapper
# -------------------------
def class_calc(cosmo_params,
               tol_background=1e-10, tol_thermo=1e-6,
               input_data=None):
    from classy import Class
    h = cosmo_params[0] / 100.0
    params = {
        'output': 'mPk',
        'omega_b': cosmo_params[1],
        'omega_cdm': cosmo_params[2],
        'h': h,
        'N_ncdm': 1,
        'N_ur': 2.046,
        'm_ncdm': cosmo_params[3],
        'tau_reio': 0.054,
        'A_s': cosmo_params[4],
        'n_s': cosmo_params[5],
        'P_k_max_h/Mpc': 10.0,
        'z_pk': 0,
        'tol_background_integration': tol_background,
        'tol_thermo_integration': tol_thermo,
    }
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    kh = np.logspace(-4, 1, 200)
    k = kh * h
    Plin = np.array([cosmo.pk_lin(ki, 0) for ki in k])
    cosmo.struct_cleanup()
    cosmo.empty()
    return pd.DataFrame({"pk": Plin * h**3})

# -------------------------
# Likelihood for powerspectrum
# -------------------------
def synthetic_gaussian(theta, like_args):
    mean = like_args["mean"]
    cov = like_args["cov"]
    diff = theta - mean
    inv_cov = np.linalg.inv(cov)
    return -0.5 * diff.T @ inv_cov @ diff

def log_prior(theta, lower=None, upper=None):
    if lower is None:
        lower = -20
    if upper is None:
        upper = 20
    if np.all((theta >= lower) & (theta <= upper)):
        return 0.0
    return -np.inf

def log_posterior(theta, like_args):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + synthetic_gaussian(theta, like_args)

# -------------------------
# Emcee runner (returns samples and logp)
# -------------------------
def run_emcee(initial_guess, loglike_func, like_args=None, n_walkers=50, n_steps=500):
    """
    returns dict with keys:
      - 'samples' : np.ndarray (N, ndim)
      - 'logp'    : np.ndarray (N,)
      - all used params returned under 'meta'
    """
    import emcee
    dim = len(initial_guess)
    np.random.seed(62)
    # Initialize walker positions near initial guess
    p0 = initial_guess + 1e-4 * np.random.randn(n_walkers, dim)

    # Run sampler
    sampler = emcee.EnsembleSampler(n_walkers, dim, loglike_func, args=(like_args,))
    sampler.run_mcmc(p0, n_steps, progress=False)

    burn_in = min(100, n_steps // 2)
    # Flatten samples and discard burn-in
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_prob = sampler.get_log_prob(discard=burn_in, flat=True)

    return {
        "samples": samples,
        "logp": log_prob,
        "meta": {"n_walkers": n_walkers, "n_steps": n_steps}
    }

# -------------------------
# Comparison helpers
# -------------------------
def compare_emcee_samples(run_output, std_output):
    """
    run_output: dict from run_emcee (contains 'samples' np.ndarray)
    std_output: dict expected to contain 'mean' (1D array) or DataFrame with mu column
    returns (error, spread)
    error: pull bias
    spread: mean of per-parameter stddevs
    """
    
    samples = run_output["samples"]
    if not np.all(np.isfinite(samples)):
        return np.nan, np.nan
    est_mean = np.mean(samples, axis=0)
    est_std = np.std(samples, axis=0)
    est_std = np.where(est_std == 0, 1e-16, est_std)

    if isinstance(std_output, dict) and "mean" in std_output:
        true_mean = np.asarray(std_output["mean"])
    elif isinstance(std_output, pd.DataFrame) and "mu" in std_output.columns:
        true_mean = std_output["mu"].to_numpy()
    else:
        raise ValueError("std_output must contain true 'mean' for emcee comparison")

    pull = (est_mean - true_mean) / (est_std)
    return float(np.mean(pull)), float(np.std(pull))
