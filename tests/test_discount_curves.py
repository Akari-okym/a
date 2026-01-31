"""
Tests for discount curve estimation and visualization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import numpy as np
import pandas as pd
import pytest
from discount_curve_utils import (
    exponential_discount,
    logit_choice_probability,
    negative_log_likelihood,
    estimate_discount_rate,
    estimate_all_participants,
    calculate_discounted_values,
    validate_data
)


def test_exponential_discount():
    """Test exponential discount function."""
    # Test single value
    assert np.isclose(exponential_discount(0, 0.1), 1.0)
    assert np.isclose(exponential_discount(10, 0.1), np.exp(-1.0))
    
    # Test array
    t = np.array([0, 10, 20])
    expected = np.array([1.0, np.exp(-1.0), np.exp(-2.0)])
    result = exponential_discount(t, 0.1)
    assert np.allclose(result, expected)


def test_logit_choice_probability():
    """Test logit choice probability function."""
    # Equal values -> 50% probability
    assert np.isclose(logit_choice_probability(100, 100), 0.5)
    
    # Delayed much better -> high probability
    assert logit_choice_probability(200, 100) > 0.7
    
    # Immediate better -> low probability
    assert logit_choice_probability(100, 200) < 0.3


def test_calculate_discounted_values():
    """Test discounted value calculation."""
    delays = np.array([0, 30, 90])
    r = 0.01
    amount_delayed = 11000
    
    values = calculate_discounted_values(delays, r, amount_delayed)
    
    # Check values are decreasing
    assert values[0] > values[1] > values[2]
    
    # Check initial value
    assert np.isclose(values[0], amount_delayed)


def test_estimate_discount_rate_normal_case():
    """Test discount rate estimation with mixed choices."""
    # Create simple test data
    data = pd.DataFrame({
        't': [7, 14, 30, 60, 90],
        'choice': [1, 1, 1, 0, 0]  # Mostly delayed for short, immediate for long
    })
    
    result = estimate_discount_rate(data)
    
    # Check result structure
    assert 'r' in result
    assert 'convergence' in result
    assert 'n_obs' in result
    assert 'prop_delayed' in result
    
    # Check reasonable values
    assert result['r'] > 0
    assert result['n_obs'] == 5
    assert 0 < result['prop_delayed'] < 1


def test_estimate_discount_rate_always_immediate():
    """Test edge case: participant always chooses immediate."""
    data = pd.DataFrame({
        't': [7, 14, 30, 60],
        'choice': [0, 0, 0, 0]
    })
    
    with pytest.warns(UserWarning):
        result = estimate_discount_rate(data, r_bounds=(1e-6, 10.0))
    
    # Should return high discount rate
    assert result['r'] == 10.0
    assert result['edge_case'] == 'always_immediate'
    assert result['prop_delayed'] == 0.0


def test_estimate_discount_rate_always_delayed():
    """Test edge case: participant always chooses delayed."""
    data = pd.DataFrame({
        't': [7, 14, 30, 60],
        'choice': [1, 1, 1, 1]
    })
    
    with pytest.warns(UserWarning):
        result = estimate_discount_rate(data, r_bounds=(1e-6, 10.0))
    
    # Should return low discount rate
    assert result['r'] == 1e-6
    assert result['edge_case'] == 'always_delayed'
    assert result['prop_delayed'] == 1.0


def test_estimate_all_participants():
    """Test estimation for multiple participants."""
    data = pd.DataFrame({
        'participant_id': ['P1', 'P1', 'P1', 'P2', 'P2', 'P2'],
        't': [7, 30, 90, 7, 30, 90],
        'choice': [1, 1, 0, 1, 0, 0]
    })
    
    results = estimate_all_participants(data)
    
    # Check result structure
    assert len(results) == 2
    assert 'participant_id' in results.columns
    assert 'r' in results.columns
    
    # Check all participants present
    assert set(results['participant_id']) == {'P1', 'P2'}


def test_validate_data_valid():
    """Test data validation with valid data."""
    data = pd.DataFrame({
        'participant_id': ['P1', 'P1', 'P2', 'P2'],
        't': [7, 30, 7, 30],
        'choice': [1, 0, 1, 1]
    })
    
    is_valid, errors = validate_data(data)
    assert is_valid
    assert len(errors) == 0


def test_validate_data_missing_columns():
    """Test data validation with missing columns."""
    data = pd.DataFrame({
        'participant_id': ['P1', 'P1'],
        't': [7, 30]
        # Missing 'choice' column
    })
    
    is_valid, errors = validate_data(data)
    assert not is_valid
    assert len(errors) > 0
    assert any('Missing required columns' in err for err in errors)


def test_validate_data_invalid_choice():
    """Test data validation with invalid choice values."""
    data = pd.DataFrame({
        'participant_id': ['P1', 'P1'],
        't': [7, 30],
        'choice': [1, 2]  # Invalid: should be 0 or 1
    })
    
    is_valid, errors = validate_data(data)
    assert not is_valid
    assert any('must contain only 0 or 1' in err for err in errors)


def test_validate_data_negative_delays():
    """Test data validation with negative delays."""
    data = pd.DataFrame({
        'participant_id': ['P1', 'P1'],
        't': [-7, 30],  # Negative delay
        'choice': [1, 0]
    })
    
    is_valid, errors = validate_data(data)
    assert not is_valid
    assert any('negative values' in err for err in errors)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
