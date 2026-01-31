#!/usr/bin/env python3
"""
Plot discount curves from binary choice survey data.

This script estimates individual discount functions from binary choice data
(immediate vs delayed) and creates a "spaghetti plot" showing all participants'
discount curves overlaid.

Usage:
    python plot_discount_curves.py <input_csv> [options]

Example:
    python plot_discount_curves.py ../data/sample_choices.csv --output discount_curves.png
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import utility functions
from discount_curve_utils import (
    estimate_all_participants,
    calculate_discounted_values,
    validate_data
)


def parse_choice_column(data, choice_col='choice', delayed_values=None, immediate_values=None):
    """
    Parse and standardize choice column to binary 0/1.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    choice_col : str
        Name of choice column
    delayed_values : list, optional
        Values that indicate "delayed" choice (e.g., ['delayed', 'later', 1])
    immediate_values : list, optional
        Values that indicate "immediate" choice (e.g., ['immediate', 'now', 0])
        
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized binary choice column
    """
    if delayed_values is None:
        delayed_values = ['delayed', 'later', 'b', 'B', 1, '1', 1.0]
    if immediate_values is None:
        immediate_values = ['immediate', 'now', 'a', 'A', 0, '0', 0.0]
    
    data = data.copy()
    
    # Convert to binary
    def map_choice(x):
        if x in delayed_values:
            return 1
        elif x in immediate_values:
            return 0
        else:
            return np.nan
    
    data[choice_col] = data[choice_col].apply(map_choice)
    
    # Check for unmapped values
    n_missing = data[choice_col].isna().sum()
    if n_missing > 0:
        print(f"Warning: {n_missing} choices could not be mapped to 0 or 1")
    
    return data


def plot_discount_curves(estimates_df, data, participant_col='participant_id',
                         amount_delayed=11000, amount_immediate=10000,
                         output_path='discount_curves.png', show=False,
                         plot_type='value', delay_unit='days'):
    """
    Create spaghetti plot of discount curves for all participants.
    
    Parameters
    ----------
    estimates_df : pd.DataFrame
        DataFrame with estimated discount rates (output of estimate_all_participants)
    data : pd.DataFrame
        Original data (used to determine delay range)
    participant_col : str
        Name of participant ID column
    amount_delayed : float
        Amount of delayed reward (default=11000 JPY)
    amount_immediate : float
        Amount of immediate reward (default=10000 JPY)
    output_path : str
        Path to save figure
    show : bool
        Whether to display figure interactively
    plot_type : str
        Type of plot: 'value' (discounted value) or 'factor' (discount factor)
    delay_unit : str
        Unit of delay (for axis label)
    """
    # Create delay range for plotting
    t_min = data['t'].min()
    t_max = data['t'].max()
    t_range = np.linspace(t_min, t_max, 100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each participant's curve
    for _, row in estimates_df.iterrows():
        participant_id = row[participant_col]
        r = row['r']
        
        if plot_type == 'value':
            # Plot discounted value
            values = calculate_discounted_values(t_range, r, amount_delayed)
            ax.plot(t_range, values, alpha=0.3, linewidth=1, color='steelblue')
        else:
            # Plot discount factor
            factors = np.exp(-r * t_range)
            ax.plot(t_range, factors, alpha=0.3, linewidth=1, color='steelblue')
    
    # Calculate and plot median curve
    median_r = estimates_df['r'].median()
    if plot_type == 'value':
        median_values = calculate_discounted_values(t_range, median_r, amount_delayed)
        ax.plot(t_range, median_values, color='darkred', linewidth=2.5, 
                label=f'Median (r={median_r:.4f})', zorder=100)
        
        # Add horizontal line for immediate option
        ax.axhline(y=amount_immediate, color='black', linestyle='--', 
                  linewidth=1.5, label=f'Immediate option ({amount_immediate} JPY)')
        
        ax.set_ylabel('Discounted Value (JPY)', fontsize=12)
        ax.set_ylim(bottom=0, top=amount_delayed * 1.1)
        
    else:
        median_factors = np.exp(-median_r * t_range)
        ax.plot(t_range, median_factors, color='darkred', linewidth=2.5, 
                label=f'Median (r={median_r:.4f})', zorder=100)
        
        ax.set_ylabel('Discount Factor D(t)', fontsize=12)
        ax.set_ylim(0, 1.05)
    
    ax.set_xlabel(f'Delay ({delay_unit})', fontsize=12)
    ax.set_title('Individual Discount Curves (Spaghetti Plot)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    # Show interactively if requested
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(
        description='Estimate and visualize discount curves from binary choice data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_discount_curves.py data.csv
  python plot_discount_curves.py data.csv --output my_plot.png --show
  python plot_discount_curves.py data.csv --participant-col ID --delay-col days
  python plot_discount_curves.py data.csv --plot-type factor

Expected CSV format:
  participant_id,t,choice
  P001,7,delayed
  P001,30,immediate
  P002,7,delayed
  ...

Choice encoding:
  The 'choice' column can use various encodings:
  - Binary: 0 (immediate) or 1 (delayed)
  - Text: "immediate"/"delayed", "now"/"later", "a"/"b"
  Both uppercase and lowercase are supported.
        """
    )
    
    parser.add_argument('input_csv', type=str,
                       help='Path to input CSV file')
    parser.add_argument('--output', '-o', type=str, default='discount_curves.png',
                       help='Output file path for plot (default: discount_curves.png)')
    parser.add_argument('--participant-col', type=str, default='participant_id',
                       help='Name of participant ID column (default: participant_id)')
    parser.add_argument('--delay-col', type=str, default='t',
                       help='Name of delay column (default: t)')
    parser.add_argument('--choice-col', type=str, default='choice',
                       help='Name of choice column (default: choice)')
    parser.add_argument('--amount-delayed', type=float, default=11000,
                       help='Amount of delayed reward (default: 11000)')
    parser.add_argument('--amount-immediate', type=float, default=10000,
                       help='Amount of immediate reward (default: 10000)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature parameter for logit choice model (default: 1.0)')
    parser.add_argument('--plot-type', type=str, choices=['value', 'factor'], default='value',
                       help='Plot discounted value or discount factor (default: value)')
    parser.add_argument('--delay-unit', type=str, default='days',
                       help='Unit of delay for axis label (default: days)')
    parser.add_argument('--show', action='store_true',
                       help='Show plot interactively')
    parser.add_argument('--save-estimates', type=str, default=None,
                       help='Save estimated parameters to CSV file')
    
    args = parser.parse_args()
    
    # Check input file exists
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_csv}")
        sys.exit(1)
    
    # Load data
    print(f"Loading data from: {args.input_csv}")
    try:
        data = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(data)} observations")
    
    # Standardize column names if needed
    col_mapping = {}
    if args.delay_col != 't':
        col_mapping[args.delay_col] = 't'
    if args.choice_col != 'choice':
        col_mapping[args.choice_col] = 'choice'
    
    if col_mapping:
        data = data.rename(columns=col_mapping)
    
    # Parse choice column
    data = parse_choice_column(data, 'choice')
    
    # Validate data
    is_valid, errors = validate_data(data, args.participant_col, 't', 'choice')
    if not is_valid:
        print("Data validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    print("Data validation passed")
    
    # Count participants
    n_participants = data[args.participant_col].nunique()
    print(f"Found {n_participants} participants")
    
    # Estimate discount rates
    print("Estimating discount rates...")
    estimates_df = estimate_all_participants(
        data,
        participant_col=args.participant_col,
        amount_delayed=args.amount_delayed,
        amount_immediate=args.amount_immediate,
        temperature=args.temperature
    )
    
    # Print summary statistics
    print("\nEstimation summary:")
    print(f"  Median discount rate: {estimates_df['r'].median():.4f}")
    print(f"  Mean discount rate: {estimates_df['r'].mean():.4f}")
    print(f"  SD discount rate: {estimates_df['r'].std():.4f}")
    print(f"  Min discount rate: {estimates_df['r'].min():.4f}")
    print(f"  Max discount rate: {estimates_df['r'].max():.4f}")
    
    n_converged = estimates_df['convergence'].sum()
    print(f"  Converged: {n_converged}/{len(estimates_df)}")
    
    n_edge_cases = estimates_df['edge_case'].notna().sum()
    if n_edge_cases > 0:
        print(f"  Edge cases (always immediate/delayed): {n_edge_cases}")
    
    # Save estimates if requested
    if args.save_estimates:
        estimates_df.to_csv(args.save_estimates, index=False)
        print(f"\nSaved estimates to: {args.save_estimates}")
    
    # Create plot
    print("\nCreating plot...")
    plot_discount_curves(
        estimates_df, data,
        participant_col=args.participant_col,
        amount_delayed=args.amount_delayed,
        amount_immediate=args.amount_immediate,
        output_path=args.output,
        show=args.show,
        plot_type=args.plot_type,
        delay_unit=args.delay_unit
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
