"""
Delay Discounting Analysis Module

This module provides tools for analyzing delay discounting data with support for:
- Japanese delay labels (e.g., "10年後", "6か月後", "3週間後", "2日後")
- Exponential discounting model with logit fitting
- Switch point analysis for nonparametric visualization
- Overlaid discount curve plots (spaghetti plots)
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Union, List, Tuple, Optional, Dict


def parse_japanese_delay(delay_str: str) -> float:
    """
    Parse Japanese delay string and convert to years.
    
    Supported formats:
    - "10年後" -> 10.0 years
    - "6か月後", "6ヶ月後", "6ヵ月後" -> 0.5 years
    - "3週間後" -> 0.0577 years (3/52)
    - "2日後" -> 0.00548 years (2/365)
    
    Args:
        delay_str: Japanese delay string
        
    Returns:
        Delay in years as float
        
    Raises:
        ValueError: If the format is not recognized
    """
    delay_str = str(delay_str).strip()
    
    # Pattern for years: 年後
    year_pattern = r'(\d+\.?\d*)年後'
    match = re.match(year_pattern, delay_str)
    if match:
        return float(match.group(1))
    
    # Pattern for months: か月後, ヶ月後, ヵ月後
    month_pattern = r'(\d+\.?\d*)[かヶヵ]月後'
    match = re.match(month_pattern, delay_str)
    if match:
        months = float(match.group(1))
        return months / 12.0
    
    # Pattern for weeks: 週間後
    week_pattern = r'(\d+\.?\d*)週間後'
    match = re.match(week_pattern, delay_str)
    if match:
        weeks = float(match.group(1))
        return weeks / 52.0
    
    # Pattern for days: 日後
    day_pattern = r'(\d+\.?\d*)日後'
    match = re.match(day_pattern, delay_str)
    if match:
        days = float(match.group(1))
        return days / 365.0
    
    raise ValueError(f"Unrecognized delay format: {delay_str}")


def parse_delay(delay: Union[str, int, float]) -> float:
    """
    Parse delay value that can be either numeric (in years) or Japanese string.
    
    Args:
        delay: Delay as numeric value (years) or Japanese string
        
    Returns:
        Delay in years as float
    """
    if isinstance(delay, (int, float)):
        return float(delay)
    
    # Try parsing as Japanese string
    try:
        return parse_japanese_delay(delay)
    except ValueError:
        pass
    
    # Try parsing as numeric string
    try:
        return float(delay)
    except ValueError:
        raise ValueError(f"Cannot parse delay value: {delay}")


def load_discounting_data(filepath: str) -> pd.DataFrame:
    """
    Load delay discounting data from CSV file.
    
    Expected CSV format:
    - participant: Participant ID
    - t: Delay (numeric in years or Japanese string like "10年後")
    - choice: Binary choice (0 = immediate, 1 = delayed)
    
    Optional columns:
    - immediate_amount: Amount of immediate reward
    - delayed_amount: Amount of delayed reward
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with parsed data including t_years column
    """
    df = pd.read_csv(filepath)
    
    # Parse delay column to numeric years
    df['t_years'] = df['t'].apply(parse_delay)
    
    # Ensure required columns exist
    required_cols = ['participant', 't', 'choice']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def exponential_discount_value(t: np.ndarray, k: float, 
                               delayed_amount: float = 1.0) -> np.ndarray:
    """
    Calculate present value using exponential discounting model.
    
    V(t) = A * exp(-k * t)
    
    Args:
        t: Delay in years
        k: Discount rate parameter
        delayed_amount: Amount of delayed reward (default: 1.0)
        
    Returns:
        Present value of delayed reward
    """
    return delayed_amount * np.exp(-k * t)


def logit_choice_probability(immediate_value: float, delayed_value: float, 
                             beta: float = 1.0) -> float:
    """
    Calculate probability of choosing delayed option using logit model.
    
    P(delayed) = 1 / (1 + exp(-beta * (delayed_value - immediate_value)))
    
    Args:
        immediate_value: Value of immediate option
        delayed_value: Value of delayed option
        beta: Temperature parameter (default: 1.0)
        
    Returns:
        Probability of choosing delayed option
    """
    utility_diff = delayed_value - immediate_value
    return 1.0 / (1.0 + np.exp(-beta * utility_diff))


def fit_exponential_discounting(t_years: np.ndarray, choices: np.ndarray,
                                immediate_amounts: Optional[np.ndarray] = None,
                                delayed_amounts: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Fit exponential discounting model with logit choice function.
    
    Args:
        t_years: Array of delays in years
        choices: Binary array of choices (0 = immediate, 1 = delayed)
        immediate_amounts: Array of immediate amounts (default: all 1.0)
        delayed_amounts: Array of delayed amounts (default: all 1.0)
        
    Returns:
        Dictionary with fitted parameters: {'k': discount_rate, 'beta': temperature, 'nll': neg_log_likelihood}
    """
    if immediate_amounts is None:
        immediate_amounts = np.ones_like(t_years)
    if delayed_amounts is None:
        delayed_amounts = np.ones_like(t_years)
    
    def negative_log_likelihood(params):
        k, beta = params
        if k < 0 or beta < 0:
            return 1e10  # Penalty for invalid parameters
        
        # Calculate present values
        delayed_values = exponential_discount_value(t_years, k, delayed_amounts)
        
        # Calculate choice probabilities
        probs = np.array([logit_choice_probability(imm, del_val, beta) 
                         for imm, del_val in zip(immediate_amounts, delayed_values)])
        
        # Avoid log(0)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        
        # Calculate negative log likelihood
        nll = -np.sum(choices * np.log(probs) + (1 - choices) * np.log(1 - probs))
        return nll
    
    # Initial guess
    x0 = [0.1, 1.0]
    
    # Optimize
    result = minimize(negative_log_likelihood, x0, method='L-BFGS-B',
                     bounds=[(1e-6, 100), (0.01, 100)])
    
    k_opt, beta_opt = result.x
    nll = result.fun
    
    return {'k': k_opt, 'beta': beta_opt, 'nll': nll}


def calculate_switch_point(t_years: np.ndarray, choices: np.ndarray) -> Tuple[float, float]:
    """
    Calculate switch point for nonparametric analysis.
    
    Under monotonic assumption, finds:
    - Largest t where participant still chose immediate (or smallest t where chose delayed)
    
    Args:
        t_years: Array of delays in years
        choices: Binary array of choices (0 = immediate, 1 = delayed)
        
    Returns:
        Tuple of (switch_point_t, implied_discount_factor)
        If all immediate choices: returns (max_t, 0.0)
        If all delayed choices: returns (0.0, 1.0)
    """
    # Sort by delay
    sorted_indices = np.argsort(t_years)
    t_sorted = t_years[sorted_indices]
    choices_sorted = choices[sorted_indices]
    
    # Find last immediate choice
    immediate_indices = np.where(choices_sorted == 0)[0]
    delayed_indices = np.where(choices_sorted == 1)[0]
    
    if len(immediate_indices) == 0:
        # All delayed choices
        return (0.0, 1.0)
    
    if len(delayed_indices) == 0:
        # All immediate choices
        return (t_sorted[-1], 0.0)
    
    # Switch point is approximately where choices transition
    last_immediate_idx = immediate_indices[-1]
    switch_t = t_sorted[last_immediate_idx]
    
    # Implied discount factor at switch point
    # Assuming equal amounts, discount factor = exp(-k*t) ≈ 0.5 at switch
    # For simplicity, use 0.5 or calculate from position
    if last_immediate_idx < len(t_sorted) - 1:
        # Interpolate between last immediate and first delayed
        first_delayed_idx = delayed_indices[0]
        if first_delayed_idx > last_immediate_idx:
            switch_t = (t_sorted[last_immediate_idx] + t_sorted[first_delayed_idx]) / 2
    
    # Implied discount factor (assuming exponential discounting)
    discount_factor = np.exp(-0.693 / switch_t) if switch_t > 0 else 1.0
    
    return (switch_t, discount_factor)


def plot_discount_curves(data: pd.DataFrame, method: str = 'exponential',
                         output_file: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 6)):
    """
    Create overlaid plot of per-participant discount curves (spaghetti plot).
    
    Args:
        data: DataFrame with columns: participant, t_years, choice
        method: 'exponential' for fitted curves or 'switch_point' for nonparametric
        output_file: Optional path to save figure
        figsize: Figure size tuple
    """
    participants = data['participant'].unique()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate time points for smooth curves
    t_plot = np.linspace(0, data['t_years'].max() * 1.1, 100)
    
    if method == 'exponential':
        # Fit exponential model for each participant
        for participant in participants:
            pdata = data[data['participant'] == participant]
            
            try:
                params = fit_exponential_discounting(
                    pdata['t_years'].values,
                    pdata['choice'].values
                )
                
                k = params['k']
                
                # Plot discount curve
                discount_values = np.exp(-k * t_plot)
                ax.plot(t_plot, discount_values, alpha=0.5, linewidth=1.5, 
                       label=f'P{participant} (k={k:.3f})')
                
            except Exception as e:
                print(f"Warning: Could not fit participant {participant}: {e}")
                continue
    
    elif method == 'switch_point':
        # Plot switch points for each participant
        switch_points = []
        for participant in participants:
            pdata = data[data['participant'] == participant]
            
            try:
                switch_t, discount_factor = calculate_switch_point(
                    pdata['t_years'].values,
                    pdata['choice'].values
                )
                switch_points.append((participant, switch_t, discount_factor))
                
                # Plot implied discount function
                if switch_t > 0:
                    k_implied = 0.693 / switch_t  # Assuming 50% point
                    discount_values = np.exp(-k_implied * t_plot)
                    ax.plot(t_plot, discount_values, alpha=0.5, linewidth=1.5,
                           label=f'P{participant} (switch={switch_t:.2f}y)')
                
            except Exception as e:
                print(f"Warning: Could not calculate switch point for participant {participant}: {e}")
                continue
    
    ax.set_xlabel('Delay (years)', fontsize=12)
    ax.set_ylabel('Discount Factor (Present Value)', fontsize=12)
    ax.set_title('Per-Participant Discount Curves (Spaghetti Plot)', fontsize=14)
    ax.set_ylim([0, 1.1])
    ax.set_xlim([0, t_plot[-1]])
    ax.grid(True, alpha=0.3)
    
    # Add legend if not too many participants
    if len(participants) <= 10:
        ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return fig, ax


def generate_synthetic_data(n_participants: int = 5,
                           staircase_delays: Optional[List[str]] = None,
                           output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Generate synthetic delay discounting data with staircase design.
    
    Args:
        n_participants: Number of participants to simulate
        staircase_delays: List of delay strings (default: standard staircase)
        output_file: Optional path to save CSV
        
    Returns:
        DataFrame with synthetic data
    """
    if staircase_delays is None:
        # Standard staircase design
        staircase_delays = [
            "10年後", "7年後", "5年後", "3年後", "1年後",
            "6か月後", "2か月後", "1か月後",
            "3週間後", "2週間後", "1週間後",
            "3日後", "2日後", "1日後"
        ]
    
    data = []
    
    for p_id in range(1, n_participants + 1):
        # Random discount rate for this participant
        k_true = np.random.uniform(0.01, 1.0)
        beta_true = np.random.uniform(0.5, 2.0)
        
        for delay_str in staircase_delays:
            t_years = parse_japanese_delay(delay_str)
            
            # Calculate present value
            delayed_value = exponential_discount_value(np.array([t_years]), k_true)[0]
            immediate_value = 1.0
            
            # Simulate choice using logit
            prob_delayed = logit_choice_probability(immediate_value, delayed_value, beta_true)
            choice = 1 if np.random.random() < prob_delayed else 0
            
            data.append({
                'participant': p_id,
                't': delay_str,
                'choice': choice,
                'immediate_amount': 1.0,
                'delayed_amount': 1.0
            })
    
    df = pd.DataFrame(data)
    
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Synthetic data saved to {output_file}")
    
    return df


if __name__ == "__main__":
    # Demo: Generate synthetic data and plot
    print("Generating synthetic delay discounting data...")
    synthetic_data = generate_synthetic_data(
        n_participants=5,
        output_file="/tmp/synthetic_discounting_data.csv"
    )
    
    print("\nLoading data and parsing delays...")
    data = load_discounting_data("/tmp/synthetic_discounting_data.csv")
    print(f"Loaded {len(data)} observations from {data['participant'].nunique()} participants")
    
    print("\nFitting exponential models...")
    for participant in data['participant'].unique():
        pdata = data[data['participant'] == participant]
        params = fit_exponential_discounting(
            pdata['t_years'].values,
            pdata['choice'].values
        )
        print(f"Participant {participant}: k={params['k']:.3f}, beta={params['beta']:.2f}")
    
    print("\nGenerating spaghetti plot...")
    fig, ax = plot_discount_curves(
        data,
        method='exponential',
        output_file="/tmp/discount_curves.png"
    )
    plt.show()
    
    print("\nDemo complete!")
