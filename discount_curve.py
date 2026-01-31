#!/usr/bin/env python3
"""
Discount Curve Estimation and Visualization

This script estimates and visualizes discount curves from intertemporal choice data.
- Uses binary choice encoding: 0 = immediate, 1 = delayed
- All delays are in integer days
- Supports Japanese labels (年/月/週/日) and English labels (y/mo/w/d)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import re


def parse_delay_to_days(delay_str):
    """
    Convert delay string to integer days.
    
    Supports formats:
    - Japanese: 年 (year), 月 (month), 週 (week), 日 (day)
    - English: y (year), mo (month), w (week), d (day)
    
    Args:
        delay_str: String like "10y", "6mo", "3w", "1d", "1年", "6月", "3週", "1日"
        
    Returns:
        Integer number of days
        
    Examples:
        >>> parse_delay_to_days("10y")
        3650
        >>> parse_delay_to_days("6mo")
        180
        >>> parse_delay_to_days("3w")
        21
        >>> parse_delay_to_days("1d")
        1
        >>> parse_delay_to_days("1年")
        365
        >>> parse_delay_to_days("6月")
        180
    """
    delay_str = str(delay_str).strip()
    
    # Japanese patterns
    if '年' in delay_str:
        match = re.search(r'(\d+(?:\.\d+)?)年', delay_str)
        if match:
            return int(float(match.group(1)) * 365)
    elif '月' in delay_str:
        match = re.search(r'(\d+(?:\.\d+)?)月', delay_str)
        if match:
            return int(float(match.group(1)) * 30)
    elif '週' in delay_str:
        match = re.search(r'(\d+(?:\.\d+)?)週', delay_str)
        if match:
            return int(float(match.group(1)) * 7)
    elif '日' in delay_str:
        match = re.search(r'(\d+(?:\.\d+)?)日', delay_str)
        if match:
            return int(float(match.group(1)))
    
    # English patterns
    match = re.search(r'(\d+(?:\.\d+)?)(y|mo|w|d)', delay_str, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        
        if unit == 'y':
            return int(value * 365)
        elif unit == 'mo':
            return int(value * 30)
        elif unit == 'w':
            return int(value * 7)
        elif unit == 'd':
            return int(value)
    
    # If it's just a number, assume it's already in days
    try:
        return int(float(delay_str))
    except ValueError:
        raise ValueError(f"Cannot parse delay: {delay_str}")


def load_choice_data(csv_path):
    """
    Load choice data from CSV file.
    
    Expected columns:
    - delay: Delay period (e.g., "10y", "6mo", "3w", "1d", "1年", etc.)
    - choice: Binary choice (0 = immediate, 1 = delayed)
    - amount_immediate: Amount for immediate option (optional)
    - amount_delayed: Amount for delayed option (optional)
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with 'delay_days' column added
    """
    df = pd.read_csv(csv_path)
    
    # Convert delay to days
    df['delay_days'] = df['delay'].apply(parse_delay_to_days)
    
    # Ensure choice is binary
    if not df['choice'].isin([0, 1]).all():
        raise ValueError("Choice column must contain only 0 (immediate) or 1 (delayed)")
    
    return df


def hyperbolic_discount_function(t, k):
    """
    Hyperbolic discount function: 1 / (1 + k*t)
    
    Args:
        t: Time delay in days (integer or array)
        k: Discount rate parameter (per day)
        
    Returns:
        Discount factor between 0 and 1
    """
    return 1.0 / (1.0 + k * t)


def exponential_discount_function(t, k):
    """
    Exponential discount function: exp(-k*t)
    
    Args:
        t: Time delay in days (integer or array)
        k: Discount rate parameter (per day)
        
    Returns:
        Discount factor between 0 and 1
    """
    return np.exp(-k * t)


def estimate_discount_rate(df, model='hyperbolic'):
    """
    Estimate discount rate from choice data using maximum likelihood.
    
    Args:
        df: DataFrame with 'delay_days' and 'choice' columns
        model: 'hyperbolic' or 'exponential'
        
    Returns:
        Estimated discount rate k (per day)
    """
    # Constants for numerical stability
    INVALID_K_PENALTY = 1e10  # Large penalty for invalid discount rates
    EPSILON = 1e-10  # Small value to prevent log(0)
    
    if model == 'hyperbolic':
        discount_func = hyperbolic_discount_function
    elif model == 'exponential':
        discount_func = exponential_discount_function
    else:
        raise ValueError(f"Unknown model: {model}")
    
    def negative_log_likelihood(k):
        if k <= 0:
            return INVALID_K_PENALTY
        
        # Calculate discount factors for each delay
        discount_factors = discount_func(df['delay_days'].values, k)
        
        # Logistic model: P(delayed) = 1 / (1 + exp(-beta * (V_delayed - V_immediate)))
        # If amounts are equal, V_delayed - V_immediate = discount_factor - 1
        # Simplified: P(delayed) = discount_factor
        
        # Calculate log likelihood
        p_delayed = np.clip(discount_factors, EPSILON, 1 - EPSILON)
        
        choices = df['choice'].values
        log_likelihood = np.sum(
            choices * np.log(p_delayed) + 
            (1 - choices) * np.log(1 - p_delayed)
        )
        
        return -log_likelihood
    
    # Initial guess: k ~ 0.001 per day (moderate discounting)
    # This corresponds to k_annual ~ 0.365 per year, a reasonable starting point
    # for typical intertemporal choice data
    result = minimize(
        negative_log_likelihood,
        x0=[0.001],
        method='L-BFGS-B',
        bounds=[(1e-6, 1.0)]
    )
    
    return result.x[0]


def plot_discount_curve(df, k, model='hyperbolic', save_path=None):
    """
    Plot the discount curve and observed choice data.
    
    Args:
        df: DataFrame with choice data
        k: Estimated discount rate (per day)
        model: 'hyperbolic' or 'exponential'
        save_path: Optional path to save the plot
    """
    if model == 'hyperbolic':
        discount_func = hyperbolic_discount_function
        model_name = "Hyperbolic"
    elif model == 'exponential':
        discount_func = exponential_discount_function
        model_name = "Exponential"
    else:
        raise ValueError(f"Unknown model: {model}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot fitted curve
    t_range = np.linspace(0, df['delay_days'].max() * 1.1, 1000)
    discount_values = discount_func(t_range, k)
    ax.plot(t_range, discount_values, 'b-', linewidth=2, 
            label=f'{model_name} (k={k:.6f} per day)')
    
    # Plot observed choices
    delayed_choices = df[df['choice'] == 1]
    immediate_choices = df[df['choice'] == 0]
    
    ax.scatter(delayed_choices['delay_days'], 
               np.ones(len(delayed_choices)) * 0.95,
               c='green', marker='o', s=100, alpha=0.6, 
               label='Chose Delayed')
    ax.scatter(immediate_choices['delay_days'], 
               np.ones(len(immediate_choices)) * 0.05,
               c='red', marker='x', s=100, alpha=0.6, 
               label='Chose Immediate')
    
    ax.set_xlabel('Delay (days)', fontsize=12)
    ax.set_ylabel('Discount Factor', fontsize=12)
    ax.set_title(f'{model_name} Discount Curve', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, df['delay_days'].max() * 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def generate_staircase_delays():
    """
    Generate standard staircase delay list.
    
    Returns:
        List of delay strings: ['10y', '7y', '5y', '3y', '1y', '6mo', '2mo', 
                                '1mo', '3w', '2w', '1w', '3d', '2d', '1d']
    """
    return ['10y', '7y', '5y', '3y', '1y', '6mo', '2mo', '1mo', 
            '3w', '2w', '1w', '3d', '2d', '1d']


def main():
    """
    Main function demonstrating usage.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python discount_curve.py <csv_file> [model]")
        print("  model: 'hyperbolic' (default) or 'exponential'")
        print("\nGenerating example CSV...")
        
        # Generate example data
        staircase = generate_staircase_delays()
        example_data = []
        
        for delay in staircase:
            delay_days = parse_delay_to_days(delay)
            # Simulate choices with hyperbolic discounting (k ~ 0.002)
            k_true = 0.002
            prob_delayed = hyperbolic_discount_function(delay_days, k_true)
            choice = 1 if np.random.rand() < prob_delayed else 0
            example_data.append({
                'delay': delay,
                'choice': choice
            })
        
        example_df = pd.DataFrame(example_data)
        example_df.to_csv('example_choices.csv', index=False)
        print("Created example_choices.csv")
        
        # Load and analyze
        df = load_choice_data('example_choices.csv')
        print(f"\nLoaded {len(df)} choice observations")
        print(f"Delay range: {df['delay_days'].min()} to {df['delay_days'].max()} days")
        
        model = 'hyperbolic'
        k = estimate_discount_rate(df, model=model)
        print(f"\nEstimated {model} discount rate: k = {k:.6f} per day")
        print(f"Equivalent annual rate: k_annual = {k * 365:.4f} per year")
        
        plot_discount_curve(df, k, model=model, save_path='discount_curve.png')
        
    else:
        csv_file = sys.argv[1]
        model = sys.argv[2] if len(sys.argv) > 2 else 'hyperbolic'
        
        # Load data
        df = load_choice_data(csv_file)
        print(f"Loaded {len(df)} choice observations")
        print(f"Delay range: {df['delay_days'].min()} to {df['delay_days'].max()} days")
        
        # Estimate discount rate
        k = estimate_discount_rate(df, model=model)
        print(f"\nEstimated {model} discount rate: k = {k:.6f} per day")
        print(f"Equivalent annual rate: k_annual = {k * 365:.4f} per year")
        
        # Plot
        plot_discount_curve(df, k, model=model, save_path='discount_curve.png')


if __name__ == '__main__':
    main()
