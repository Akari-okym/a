"""
Unit tests for delay discounting analysis module
"""

import unittest
import numpy as np
import pandas as pd
import os
import tempfile
from delay_discounting import (
    parse_japanese_delay,
    parse_delay,
    load_discounting_data,
    exponential_discount_value,
    logit_choice_probability,
    fit_exponential_discounting,
    calculate_switch_point,
    generate_synthetic_data,
    plot_discount_curves
)


class TestDelayParsing(unittest.TestCase):
    """Test delay parsing functions"""
    
    def test_parse_years(self):
        """Test parsing of year delays"""
        self.assertAlmostEqual(parse_japanese_delay("10年後"), 10.0)
        self.assertAlmostEqual(parse_japanese_delay("7年後"), 7.0)
        self.assertAlmostEqual(parse_japanese_delay("1年後"), 1.0)
    
    def test_parse_months(self):
        """Test parsing of month delays"""
        self.assertAlmostEqual(parse_japanese_delay("6か月後"), 0.5)
        self.assertAlmostEqual(parse_japanese_delay("6ヶ月後"), 0.5)
        self.assertAlmostEqual(parse_japanese_delay("6ヵ月後"), 0.5)
        self.assertAlmostEqual(parse_japanese_delay("12か月後"), 1.0)
        self.assertAlmostEqual(parse_japanese_delay("1か月後"), 1/12, places=5)
    
    def test_parse_weeks(self):
        """Test parsing of week delays"""
        self.assertAlmostEqual(parse_japanese_delay("3週間後"), 3/52, places=5)
        self.assertAlmostEqual(parse_japanese_delay("2週間後"), 2/52, places=5)
        self.assertAlmostEqual(parse_japanese_delay("1週間後"), 1/52, places=5)
    
    def test_parse_days(self):
        """Test parsing of day delays"""
        self.assertAlmostEqual(parse_japanese_delay("3日後"), 3/365, places=5)
        self.assertAlmostEqual(parse_japanese_delay("2日後"), 2/365, places=5)
        self.assertAlmostEqual(parse_japanese_delay("1日後"), 1/365, places=5)
    
    def test_parse_numeric(self):
        """Test parsing of numeric delays"""
        self.assertAlmostEqual(parse_delay(10), 10.0)
        self.assertAlmostEqual(parse_delay(0.5), 0.5)
        self.assertAlmostEqual(parse_delay("5.5"), 5.5)
    
    def test_parse_invalid(self):
        """Test handling of invalid formats"""
        with self.assertRaises(ValueError):
            parse_japanese_delay("invalid")
        
        with self.assertRaises(ValueError):
            parse_delay("not a delay")


class TestDiscountingModels(unittest.TestCase):
    """Test discounting model functions"""
    
    def test_exponential_discount_value(self):
        """Test exponential discounting calculation"""
        # At t=0, value should equal delayed amount
        self.assertAlmostEqual(
            exponential_discount_value(np.array([0]), 0.1, 1.0)[0],
            1.0
        )
        
        # At very large t, value should approach 0
        self.assertLess(
            exponential_discount_value(np.array([100]), 0.1, 1.0)[0],
            0.01
        )
        
        # Check specific value
        # V(1) = exp(-0.693) ≈ 0.5 when k=0.693
        self.assertAlmostEqual(
            exponential_discount_value(np.array([1]), 0.693, 1.0)[0],
            0.5,
            places=2
        )
    
    def test_logit_choice_probability(self):
        """Test logit choice probability"""
        # When delayed value > immediate, prob should be > 0.5
        self.assertGreater(
            logit_choice_probability(0.4, 0.6, beta=1.0),
            0.5
        )
        
        # When values equal, prob should be 0.5
        self.assertAlmostEqual(
            logit_choice_probability(0.5, 0.5, beta=1.0),
            0.5
        )
        
        # When immediate value > delayed, prob should be < 0.5
        self.assertLess(
            logit_choice_probability(0.6, 0.4, beta=1.0),
            0.5
        )
    
    def test_fit_exponential_discounting(self):
        """Test model fitting"""
        # Create simple test data
        # High k means steep discounting (prefer immediate)
        t_years = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        choices = np.array([0, 0, 0, 0, 1])  # Mostly immediate
        
        params = fit_exponential_discounting(t_years, choices)
        
        # Should return valid parameters
        self.assertIn('k', params)
        self.assertIn('beta', params)
        self.assertIn('nll', params)
        
        # k should be positive
        self.assertGreater(params['k'], 0)
        self.assertGreater(params['beta'], 0)
    
    def test_calculate_switch_point(self):
        """Test switch point calculation"""
        # All immediate choices
        t_years = np.array([0.1, 0.5, 1.0, 2.0])
        choices = np.array([0, 0, 0, 0])
        switch_t, discount_factor = calculate_switch_point(t_years, choices)
        self.assertEqual(switch_t, 2.0)
        self.assertEqual(discount_factor, 0.0)
        
        # All delayed choices
        choices = np.array([1, 1, 1, 1])
        switch_t, discount_factor = calculate_switch_point(t_years, choices)
        self.assertEqual(switch_t, 0.0)
        self.assertEqual(discount_factor, 1.0)
        
        # Mixed choices
        choices = np.array([0, 0, 1, 1])
        switch_t, discount_factor = calculate_switch_point(t_years, choices)
        self.assertGreater(switch_t, 0)
        self.assertLess(switch_t, 2.0)


class TestDataLoading(unittest.TestCase):
    """Test data loading and CSV handling"""
    
    def test_load_discounting_data(self):
        """Test loading CSV with mixed delay formats"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("participant,t,choice\n")
            f.write("1,10年後,1\n")
            f.write("1,1年後,0\n")
            f.write("2,6か月後,1\n")
            f.write("2,1週間後,0\n")
            temp_path = f.name
        
        try:
            # Import the actual function from the module
            from delay_discounting import load_discounting_data as load_data_func
            
            # Load data
            df = load_data_func(temp_path)
            
            # Check structure
            self.assertIn('t_years', df.columns)
            self.assertEqual(len(df), 4)
            
            # Check parsing
            self.assertAlmostEqual(df.iloc[0]['t_years'], 10.0)
            self.assertAlmostEqual(df.iloc[2]['t_years'], 0.5)
            
        finally:
            os.unlink(temp_path)
    
    def test_load_numeric_delays(self):
        """Test loading CSV with numeric delays"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("participant,t,choice\n")
            f.write("1,10.0,1\n")
            f.write("1,1.0,0\n")
            f.write("2,0.5,1\n")
            temp_path = f.name
        
        try:
            # Import the actual function from the module
            from delay_discounting import load_discounting_data as load_data_func
            
            df = load_data_func(temp_path)
            self.assertEqual(len(df), 3)
            self.assertAlmostEqual(df.iloc[0]['t_years'], 10.0)
            self.assertAlmostEqual(df.iloc[2]['t_years'], 0.5)
        finally:
            os.unlink(temp_path)


class TestSyntheticData(unittest.TestCase):
    """Test synthetic data generation"""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation"""
        df = generate_synthetic_data(n_participants=3)
        
        # Check structure
        self.assertIn('participant', df.columns)
        self.assertIn('t', df.columns)
        self.assertIn('choice', df.columns)
        
        # Check we have data for all participants
        self.assertEqual(df['participant'].nunique(), 3)
        
        # Check choices are binary
        self.assertTrue(set(df['choice'].unique()).issubset({0, 1}))
    
    def test_custom_staircase(self):
        """Test with custom staircase delays"""
        custom_delays = ["5年後", "1年後", "6か月後"]
        df = generate_synthetic_data(
            n_participants=2,
            staircase_delays=custom_delays
        )
        
        # Should have n_participants * len(custom_delays) rows
        self.assertEqual(len(df), 2 * 3)
        
        # Check all delays are present
        self.assertEqual(set(df['t'].unique()), set(custom_delays))


class TestPlotting(unittest.TestCase):
    """Test plotting functions"""
    
    def test_plot_exponential_curves(self):
        """Test exponential curve plotting"""
        # Generate small synthetic dataset
        df = generate_synthetic_data(n_participants=2)
        # Add t_years column
        df['t_years'] = df['t'].apply(parse_delay)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            # Should not raise errors
            fig, ax = plot_discount_curves(
                df,
                method='exponential',
                output_file=temp_path
            )
            
            # Check file was created
            self.assertTrue(os.path.exists(temp_path))
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_plot_switch_points(self):
        """Test switch point plotting"""
        df = generate_synthetic_data(n_participants=2)
        # Add t_years column
        df['t_years'] = df['t'].apply(parse_delay)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            fig, ax = plot_discount_curves(
                df,
                method='switch_point',
                output_file=temp_path
            )
            
            self.assertTrue(os.path.exists(temp_path))
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
