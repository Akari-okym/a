#!/usr/bin/env python3
"""
Example script demonstrating delay discounting analysis with staircase design.

This script:
1. Generates synthetic data with Japanese delay labels
2. Fits exponential discounting models
3. Creates overlaid discount curve plots (spaghetti plots)
4. Demonstrates both parametric and nonparametric approaches
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from delay_discounting import (
    generate_synthetic_data,
    load_discounting_data,
    fit_exponential_discounting,
    calculate_switch_point,
    plot_discount_curves
)
import matplotlib.pyplot as plt


def main():
    """Run example analysis"""
    
    print("=" * 60)
    print("Delay Discounting Analysis - Example")
    print("=" * 60)
    
    # Step 1: Generate synthetic data with staircase design
    print("\n1. Generating synthetic data with staircase design...")
    print("   Delays: 10y, 7y, 5y, 3y, 1y, 6mo, 2mo, 1mo, 3w, 2w, 1w, 3d, 2d, 1d")
    
    synthetic_data = generate_synthetic_data(
        n_participants=5,
        output_file="example_data.csv"
    )
    
    print(f"   Generated {len(synthetic_data)} observations from {synthetic_data['participant'].nunique()} participants")
    
    # Step 2: Load and parse data
    print("\n2. Loading data and parsing Japanese delay labels...")
    data = load_discounting_data("example_data.csv")
    
    print("\n   Sample data:")
    print(data.head(10).to_string(index=False))
    
    # Step 3: Fit exponential models
    print("\n3. Fitting exponential discounting models per participant...")
    print("\n   Participant | k (discount rate) | beta (temperature) | NLL")
    print("   " + "-" * 60)
    
    for participant in sorted(data['participant'].unique()):
        pdata = data[data['participant'] == participant]
        params = fit_exponential_discounting(
            pdata['t_years'].values,
            pdata['choice'].values
        )
        print(f"   {participant:11} | {params['k']:17.4f} | {params['beta']:18.2f} | {params['nll']:6.2f}")
    
    # Step 4: Calculate switch points
    print("\n4. Calculating switch points (nonparametric analysis)...")
    print("\n   Participant | Switch Point (years) | Implied Discount Factor")
    print("   " + "-" * 60)
    
    for participant in sorted(data['participant'].unique()):
        pdata = data[data['participant'] == participant]
        switch_t, discount_factor = calculate_switch_point(
            pdata['t_years'].values,
            pdata['choice'].values
        )
        print(f"   {participant:11} | {switch_t:20.3f} | {discount_factor:23.3f}")
    
    # Step 5: Create spaghetti plots
    print("\n5. Creating overlaid discount curve plots...")
    
    # Exponential method
    print("   a) Exponential model fits...")
    fig1, ax1 = plot_discount_curves(
        data,
        method='exponential',
        output_file="discount_curves_exponential.png"
    )
    
    # Switch point method
    print("   b) Switch point analysis...")
    fig2, ax2 = plot_discount_curves(
        data,
        method='switch_point',
        output_file="discount_curves_switchpoint.png"
    )
    
    print("\n6. Analysis complete!")
    print("   Output files:")
    print("   - example_data.csv: Synthetic data with Japanese delay labels")
    print("   - discount_curves_exponential.png: Spaghetti plot (exponential fits)")
    print("   - discount_curves_switchpoint.png: Spaghetti plot (switch points)")
    
    # Display plots
    print("\n7. Displaying plots...")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
