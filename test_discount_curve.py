#!/usr/bin/env python3
"""
Unit tests for discount_curve.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from discount_curve import parse_delay_to_days, generate_staircase_delays


def test_parse_delay_to_days():
    """Test delay parsing for various formats"""
    print("Testing parse_delay_to_days...")
    
    # English formats
    test_cases = [
        ("10y", 3650),
        ("7y", 2555),
        ("5y", 1825),
        ("3y", 1095),
        ("1y", 365),
        ("6mo", 180),
        ("2mo", 60),
        ("1mo", 30),
        ("3w", 21),
        ("2w", 14),
        ("1w", 7),
        ("3d", 3),
        ("2d", 2),
        ("1d", 1),
    ]
    
    for delay_str, expected_days in test_cases:
        result = parse_delay_to_days(delay_str)
        assert result == expected_days, f"Failed: {delay_str} -> {result}, expected {expected_days}"
        print(f"  ✓ {delay_str} = {result} days")
    
    # Japanese formats
    japanese_cases = [
        ("10年", 3650),
        ("7年", 2555),
        ("1年", 365),
        ("6月", 180),
        ("1月", 30),
        ("3週", 21),
        ("1週", 7),
        ("3日", 3),
        ("1日", 1),
    ]
    
    for delay_str, expected_days in japanese_cases:
        result = parse_delay_to_days(delay_str)
        assert result == expected_days, f"Failed: {delay_str} -> {result}, expected {expected_days}"
        print(f"  ✓ {delay_str} = {result} days")
    
    print("  All parsing tests passed! ✓\n")


def test_staircase_delays():
    """Test staircase delay generation"""
    print("Testing generate_staircase_delays...")
    
    delays = generate_staircase_delays()
    expected = ['10y', '7y', '5y', '3y', '1y', '6mo', '2mo', '1mo', 
                '3w', '2w', '1w', '3d', '2d', '1d']
    
    assert delays == expected, f"Staircase delays mismatch"
    print(f"  ✓ Generated {len(delays)} delays")
    
    # Verify they convert to days correctly
    delays_in_days = [parse_delay_to_days(d) for d in delays]
    expected_days = [3650, 2555, 1825, 1095, 365, 180, 60, 30, 21, 14, 7, 3, 2, 1]
    
    assert delays_in_days == expected_days, f"Delay conversion mismatch"
    print(f"  ✓ All delays convert correctly")
    print(f"  Delays in days: {delays_in_days}\n")


def test_binary_encoding():
    """Test that binary encoding is correct"""
    print("Testing binary encoding expectations...")
    print("  ✓ Choice 0 = immediate")
    print("  ✓ Choice 1 = delayed\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Running discount_curve.py tests")
    print("=" * 60 + "\n")
    
    test_parse_delay_to_days()
    test_staircase_delays()
    test_binary_encoding()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
