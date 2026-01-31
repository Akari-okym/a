"""
Utility functions for estimating and visualizing discount functions from binary choice data.

This module implements exponential discounting with logistic choice rule to estimate
per-participant discount rates from binary choice data (immediate vs delayed).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
import warnings


def exponential_discount(t, r):
    """
    Calculate exponential discount factor.
    
    Parameters
    ----------
    t : float or array-like
        Time delay(s)
    r : float
        Discount rate parameter (r >= 0)
        
    Returns
    -------
    float or array
        Discount factor(s) D(t) = exp(-r*t)
    """
    return np.exp(-r * t)


def logit_choice_probability(v_delayed, v_immediate, temperature=1.0):
    """
    Calculate probability of choosing delayed option using logistic function.
    
    Parameters
    ----------
    v_delayed : float or array-like
        Value of delayed option
    v_immediate : float or array-like
        Value of immediate option
    temperature : float, optional
        Temperature parameter for logit function (default=1.0)
        Higher values make choices more random
        
    Returns
    -------
    float or array
        Probability of choosing delayed option
    """
    utility_diff = (v_delayed - v_immediate) / temperature
    return expit(utility_diff)


def negative_log_likelihood(r, delays, choices, amount_delayed, amount_immediate, 
                           temperature=1.0):
    """
    Calculate negative log-likelihood for exponential discount model.
    
    Parameters
    ----------
    r : float
        Discount rate parameter
    delays : array-like
        Array of time delays
    choices : array-like
        Binary choices (1 = delayed, 0 = immediate)
    amount_delayed : float
        Amount of delayed reward
    amount_immediate : float
        Amount of immediate reward
    temperature : float, optional
        Temperature parameter for logit function
        
    Returns
    -------
    float
        Negative log-likelihood
    """
    # Calculate discount factors
    discount_factors = exponential_discount(delays, r)
    
    # Calculate discounted value of delayed option
    v_delayed = amount_delayed * discount_factors
    
    # Calculate probability of choosing delayed
    p_delayed = logit_choice_probability(v_delayed, amount_immediate, temperature)
    
    # Clip probabilities to avoid log(0)
    p_delayed = np.clip(p_delayed, 1e-10, 1 - 1e-10)
    
    # Calculate log-likelihood
    log_likelihood = np.sum(
        choices * np.log(p_delayed) + (1 - choices) * np.log(1 - p_delayed)
    )
    
    return -log_likelihood


def estimate_discount_rate(participant_data, amount_delayed=11000, amount_immediate=10000,
                          temperature=1.0, r_bounds=(1e-6, 10.0)):
    """
    Estimate discount rate for a single participant using maximum likelihood.
    
    Parameters
    ----------
    participant_data : pd.DataFrame
        DataFrame with columns 't' (delay) and 'choice' (1=delayed, 0=immediate)
    amount_delayed : float, optional
        Amount of delayed reward (default=11000 JPY)
    amount_immediate : float, optional
        Amount of immediate reward (default=10000 JPY)
    temperature : float, optional
        Temperature parameter for logit function (default=1.0)
    r_bounds : tuple, optional
        Bounds for discount rate parameter (default=(1e-6, 10.0))
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'r': estimated discount rate
        - 'convergence': whether optimization converged
        - 'n_obs': number of observations
        - 'prop_delayed': proportion of delayed choices
    """
    delays = participant_data['t'].values
    choices = participant_data['choice'].values
    
    n_obs = len(choices)
    prop_delayed = np.mean(choices)
    
    # Handle edge cases
    if prop_delayed == 0:
        # Always chose immediate -> high discount rate
        warnings.warn("Participant always chose immediate. Setting r to upper bound.")
        return {
            'r': r_bounds[1],
            'convergence': False,
            'n_obs': n_obs,
            'prop_delayed': prop_delayed,
            'edge_case': 'always_immediate'
        }
    elif prop_delayed == 1:
        # Always chose delayed -> low discount rate
        warnings.warn("Participant always chose delayed. Setting r to lower bound.")
        return {
            'r': r_bounds[0],
            'convergence': False,
            'n_obs': n_obs,
            'prop_delayed': prop_delayed,
            'edge_case': 'always_delayed'
        }
    
    # Optimize
    result = minimize_scalar(
        negative_log_likelihood,
        bounds=r_bounds,
        method='bounded',
        args=(delays, choices, amount_delayed, amount_immediate, temperature)
    )
    
    return {
        'r': result.x,
        'convergence': result.success,
        'n_obs': n_obs,
        'prop_delayed': prop_delayed,
        'edge_case': None
    }


def estimate_all_participants(data, participant_col='participant_id',
                             amount_delayed=11000, amount_immediate=10000,
                             temperature=1.0, r_bounds=(1e-6, 10.0)):
    """
    Estimate discount rates for all participants in dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns: participant_col, 't', 'choice'
    participant_col : str, optional
        Name of participant ID column (default='participant_id')
    amount_delayed : float, optional
        Amount of delayed reward (default=11000 JPY)
    amount_immediate : float, optional
        Amount of immediate reward (default=10000 JPY)
    temperature : float, optional
        Temperature parameter for logit function (default=1.0)
    r_bounds : tuple, optional
        Bounds for discount rate parameter (default=(1e-6, 10.0))
        
    Returns
    -------
    pd.DataFrame
        DataFrame with participant estimates (one row per participant)
    """
    results = []
    
    for participant_id, group in data.groupby(participant_col):
        estimate = estimate_discount_rate(
            group, amount_delayed, amount_immediate, temperature, r_bounds
        )
        estimate[participant_col] = participant_id
        results.append(estimate)
    
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    cols = [participant_col, 'r', 'convergence', 'n_obs', 'prop_delayed', 'edge_case']
    results_df = results_df[cols]
    
    return results_df


def calculate_discounted_values(delays, r, amount_delayed=11000):
    """
    Calculate discounted values for given delays and discount rate.
    
    Parameters
    ----------
    delays : array-like
        Time delays
    r : float
        Discount rate parameter
    amount_delayed : float, optional
        Amount of delayed reward (default=11000 JPY)
        
    Returns
    -------
    array
        Discounted values V(t) = amount_delayed * exp(-r*t)
    """
    discount_factors = exponential_discount(delays, r)
    return amount_delayed * discount_factors


def validate_data(data, participant_col='participant_id', delay_col='t', choice_col='choice'):
    """
    Validate input data format and provide helpful error messages.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame to validate
    participant_col : str
        Expected participant ID column name
    delay_col : str
        Expected delay column name
    choice_col : str
        Expected choice column name
        
    Returns
    -------
    tuple
        (is_valid, error_messages)
    """
    errors = []
    
    # Check required columns
    required_cols = [participant_col, delay_col, choice_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors
    
    # Check for missing values
    for col in required_cols:
        if data[col].isna().any():
            errors.append(f"Column '{col}' contains missing values")
    
    # Check delay column is numeric and non-negative
    if not pd.api.types.is_numeric_dtype(data[delay_col]):
        errors.append(f"Delay column '{delay_col}' must be numeric")
    elif (data[delay_col] < 0).any():
        errors.append(f"Delay column '{delay_col}' contains negative values")
    
    # Check choice column is binary
    unique_choices = data[choice_col].unique()
    if not set(unique_choices).issubset({0, 1}):
        errors.append(
            f"Choice column '{choice_col}' must contain only 0 or 1. "
            f"Found: {unique_choices}"
        )
    
    # Check each participant has multiple observations
    obs_per_participant = data.groupby(participant_col).size()
    if (obs_per_participant < 2).any():
        n_insufficient = (obs_per_participant < 2).sum()
        errors.append(
            f"{n_insufficient} participant(s) have fewer than 2 observations. "
            "At least 2 observations per participant are recommended."
        )
    
    is_valid = len(errors) == 0
    return is_valid, errors
