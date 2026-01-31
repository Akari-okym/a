"""
Generate synthetic binary choice data for testing discount curve estimation.

This script creates realistic synthetic data where participants make choices
between immediate and delayed rewards according to exponential discounting.
"""

import numpy as np
import pandas as pd
from scipy.special import expit


def generate_synthetic_data(n_participants=20, delays=[7, 14, 30, 60, 90, 180],
                           amount_delayed=11000, amount_immediate=10000,
                           r_mean=0.02, r_sd=0.01, temperature=1.0,
                           seed=42):
    """
    Generate synthetic binary choice data.
    
    Parameters
    ----------
    n_participants : int
        Number of participants
    delays : list
        List of delay values (e.g., days)
    amount_delayed : float
        Amount of delayed reward
    amount_immediate : float
        Amount of immediate reward
    r_mean : float
        Mean discount rate across participants
    r_sd : float
        Standard deviation of discount rates
    temperature : float
        Temperature parameter for logistic choice function
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: participant_id, t, choice
    dict
        Dictionary with true parameters for each participant
    """
    np.random.seed(seed)
    
    data = []
    true_params = {}
    
    for i in range(n_participants):
        participant_id = f"P{i+1:03d}"
        
        # Generate true discount rate (log-normal to ensure positive)
        r_true = np.random.lognormal(np.log(r_mean) - 0.5 * r_sd**2, r_sd)
        true_params[participant_id] = {'r': r_true}
        
        for t in delays:
            # Calculate discount factor
            discount_factor = np.exp(-r_true * t)
            
            # Calculate discounted value of delayed option
            v_delayed = amount_delayed * discount_factor
            
            # Calculate probability of choosing delayed
            utility_diff = (v_delayed - amount_immediate) / temperature
            p_delayed = expit(utility_diff)
            
            # Make choice
            choice = 1 if np.random.random() < p_delayed else 0
            
            data.append({
                'participant_id': participant_id,
                't': t,
                'choice': choice
            })
    
    df = pd.DataFrame(data)
    return df, true_params


if __name__ == '__main__':
    # Generate sample data with realistic individual variation
    # Use default temperature (1.0) which matches estimation default
    dfs = []
    
    # Very patient group - low discount rates (strong preference for delayed)
    df1, _ = generate_synthetic_data(
        n_participants=10, delays=[7, 14, 30, 60, 90, 180],
        r_mean=0.001, r_sd=0.0005, temperature=1.0, seed=42
    )
    dfs.append(df1)
    
    # Moderate patience group
    df2, _ = generate_synthetic_data(
        n_participants=10, delays=[7, 14, 30, 60, 90, 180],
        r_mean=0.003, r_sd=0.001, temperature=1.0, seed=43
    )
    # Rename participants to avoid conflicts
    df2['participant_id'] = df2['participant_id'].str.replace('P', 'M')
    dfs.append(df2)
    
    # Less patient group - higher discount rates
    df3, _ = generate_synthetic_data(
        n_participants=10, delays=[7, 14, 30, 60, 90, 180],
        r_mean=0.006, r_sd=0.002, temperature=1.0, seed=44
    )
    # Rename participants to avoid conflicts
    df3['participant_id'] = df3['participant_id'].str.replace('P', 'I')
    dfs.append(df3)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Save to CSV
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, 'sample_choices.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} observations for {df['participant_id'].nunique()} participants")
    print(f"Saved to: {output_path}")
    
    # Print first few rows
    print("\nFirst few rows:")
    print(df.head(10))
    
    # Print some statistics
    print("\nChoice distribution by delay:")
    print(df.groupby('t')['choice'].agg(['mean', 'count']))
