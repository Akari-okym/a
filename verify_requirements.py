#!/usr/bin/env python3
"""
Verification script to ensure all problem statement requirements are met.
"""

import os
import sys
from delay_discounting import *
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def test_requirement(name, func):
    """Helper to run and report requirement tests"""
    try:
        func()
        print(f"✅ {name}")
        return True
    except Exception as e:
        print(f"❌ {name}: {e}")
        return False

def req1_japanese_parsing():
    """Robust parsing/conversion of delay labels"""
    test_cases = [
        ("10年後", 10.0),
        ("6か月後", 0.5),
        ("3週間後", 3/52),
        ("2日後", 2/365)
    ]
    for label, expected_years in test_cases:
        result = parse_japanese_delay(label)
        assert abs(result - expected_years) < 0.001, f"{label} parsed incorrectly"

def req2_csv_input():
    """CSV input where t may be numeric or string label"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("participant,t,choice\n")
        f.write("1,10年後,1\n")
        f.write("1,5.0,0\n")
        temp_path = f.name
    
    try:
        df = load_discounting_data(temp_path)
        assert len(df) == 2
        assert df.iloc[0]['t_years'] == 10.0
        assert df.iloc[1]['t_years'] == 5.0
    finally:
        os.unlink(temp_path)

def req3_nonparametric_viz():
    """Nonparametric visualization with switch points"""
    df = generate_synthetic_data(n_participants=2)
    df['t_years'] = df['t'].apply(parse_delay)
    
    # Test switch point calculation
    p1_data = df[df['participant'] == 1]
    switch_t, discount_factor = calculate_switch_point(
        p1_data['t_years'].values,
        p1_data['choice'].values
    )
    assert switch_t >= 0
    assert 0 <= discount_factor <= 1
    
    # Test plotting
    fig, ax = plot_discount_curves(df, method='switch_point')
    plt.close(fig)

def req4_exponential_fitting():
    """Exponential + logit per-participant fitting"""
    df = generate_synthetic_data(n_participants=3)
    df['t_years'] = df['t'].apply(parse_delay)
    
    for participant in df['participant'].unique():
        pdata = df[df['participant'] == participant]
        params = fit_exponential_discounting(
            pdata['t_years'].values,
            pdata['choice'].values
        )
        assert 'k' in params
        assert 'beta' in params
        assert params['k'] >= 0
        assert params['beta'] >= 0

def req5_staircase_design():
    """Staircase design: 10y, 7y, 5y, 3y, 1y, 6mo, 2mo, 1mo, 3w, 2w, 1w, 3d, 2d, 1d"""
    standard_staircase = [
        "10年後", "7年後", "5年後", "3年後", "1年後",
        "6か月後", "2か月後", "1か月後",
        "3週間後", "2週間後", "1週間後",
        "3日後", "2日後", "1日後"
    ]
    
    # Generate data with standard staircase
    df = generate_synthetic_data(n_participants=2, staircase_delays=standard_staircase)
    
    # Verify all delays are present
    assert set(df['t'].unique()) == set(standard_staircase)
    assert len(df) == 2 * len(standard_staircase)

def req6_spaghetti_plot():
    """Overlaid plot of per-participant discount curves"""
    df = generate_synthetic_data(n_participants=4)
    df['t_years'] = df['t'].apply(parse_delay)
    
    # Test exponential method
    fig1, ax1 = plot_discount_curves(df, method='exponential')
    assert fig1 is not None
    assert ax1 is not None
    plt.close(fig1)
    
    # Test switch point method
    fig2, ax2 = plot_discount_curves(df, method='switch_point')
    assert fig2 is not None
    assert ax2 is not None
    plt.close(fig2)

def req7_documentation():
    """README and documentation"""
    assert os.path.exists('README.md')
    with open('README.md', 'r') as f:
        content = f.read()
        assert 'Japanese' in content or '日本' in content
        assert 'CSV' in content
        assert 'exponential' in content.lower()
        assert 'switch' in content.lower()

def req8_synthetic_example():
    """Synthetic example demonstrating staircase"""
    assert os.path.exists('example.py')
    
    # Run example and check it completes
    import subprocess
    result = subprocess.run(
        [sys.executable, 'example.py'],
        capture_output=True,
        timeout=30
    )
    assert result.returncode == 0, f"Example script failed: {result.stderr}"

def req9_unit_tests():
    """Unit tests included"""
    assert os.path.exists('test_delay_discounting.py')
    
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='test_delay_discounting.py')
    runner = unittest.TextTestRunner(stream=open('/dev/null', 'w'), verbosity=0)
    result = runner.run(suite)
    assert result.wasSuccessful(), f"Unit tests failed"
    assert result.testsRun >= 10, "Not enough tests"

def main():
    print("=" * 70)
    print("VERIFYING PROBLEM STATEMENT REQUIREMENTS")
    print("=" * 70)
    print()
    
    requirements = [
        ("Robust Japanese delay label parsing", req1_japanese_parsing),
        ("CSV input with numeric/string delays", req2_csv_input),
        ("Nonparametric visualization (switch points)", req3_nonparametric_viz),
        ("Exponential + logit fitting per participant", req4_exponential_fitting),
        ("Standard staircase design support", req5_staircase_design),
        ("Spaghetti plot overlay of discount curves", req6_spaghetti_plot),
        ("README and documentation", req7_documentation),
        ("Synthetic example demonstrating staircase", req8_synthetic_example),
        ("Unit tests included", req9_unit_tests),
    ]
    
    results = []
    for name, func in requirements:
        results.append(test_requirement(name, func))
    
    print()
    print("=" * 70)
    if all(results):
        print("✅ ALL REQUIREMENTS MET")
        print("=" * 70)
        print()
        print("Acceptance Criteria:")
        print("✅ Script runs on sample data with Japanese delay labels")
        print("✅ Outputs spaghetti overlay plot")
        print("✅ Unit tests and synthetic demo included")
        print()
        print("Implementation is complete and ready for use!")
        return 0
    else:
        print(f"❌ SOME REQUIREMENTS NOT MET ({sum(results)}/{len(results)} passed)")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
