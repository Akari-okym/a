# Delay Discounting Analysis

A Python implementation for analyzing delay discounting data with support for Japanese delay labels and staircase-style experimental designs.

## Features

- **Japanese Delay Label Parsing**: Automatically parse delay strings like "10年後" (10 years), "6か月後" (6 months), "3週間後" (3 weeks), "2日後" (2 days)
- **Flexible Input**: Support both numeric delays (in years) and Japanese string labels in CSV files
- **Exponential Discounting Model**: Fit exponential discounting with logit choice function per participant
- **Nonparametric Analysis**: Calculate switch points for model-free visualization
- **Spaghetti Plots**: Create overlaid discount curve plots showing all participants
- **Synthetic Data Generation**: Built-in tools for generating example datasets

## Installation

### Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- SciPy

Install dependencies:

```bash
pip install numpy pandas matplotlib scipy
```

## Quick Start

Run the example script to see the analysis in action:

```bash
python example.py
```

This will:
1. Generate synthetic data with the standard staircase design
2. Fit exponential discounting models
3. Calculate switch points
4. Create spaghetti plots showing discount curves for all participants

## Usage

### Basic Analysis

```python
from delay_discounting import (
    load_discounting_data,
    fit_exponential_discounting,
    plot_discount_curves
)

# Load data from CSV
data = load_discounting_data("mydata.csv")

# Fit model for a participant
participant_data = data[data['participant'] == 1]
params = fit_exponential_discounting(
    participant_data['t_years'].values,
    participant_data['choice'].values
)

print(f"Discount rate k = {params['k']:.3f}")

# Create spaghetti plot
plot_discount_curves(data, method='exponential', output_file='curves.png')
```

### CSV Data Format

Your CSV file should have the following columns:

- `participant`: Participant ID (numeric or string)
- `t`: Delay (numeric in years OR Japanese string)
- `choice`: Binary choice (0 = immediate, 1 = delayed)

Optional columns:
- `immediate_amount`: Amount of immediate reward (default: 1.0)
- `delayed_amount`: Amount of delayed reward (default: 1.0)

#### Example CSV with Japanese Labels:

```csv
participant,t,choice
1,10年後,1
1,7年後,1
1,5年後,0
1,3年後,0
1,1年後,0
1,6か月後,0
1,2か月後,0
1,1か月後,0
```

#### Example CSV with Numeric Delays:

```csv
participant,t,choice
1,10.0,1
1,7.0,1
1,5.0,0
1,3.0,0
1,1.0,0
1,0.5,0
```

### Supported Delay Formats

The parser recognizes the following Japanese delay formats:

- **Years**: `10年後`, `7年後`, `5年後`, `3年後`, `1年後`
- **Months**: `6か月後`, `2か月後`, `1か月後` (also supports `ヶ月後` and `ヵ月後`)
- **Weeks**: `3週間後`, `2週間後`, `1週間後`
- **Days**: `3日後`, `2日後`, `1日後`

All delays are internally converted to years for modeling.

### Standard Staircase Design

The standard staircase includes 14 delays:

- Long delays: 10y, 7y, 5y, 3y, 1y
- Medium delays: 6mo, 2mo, 1mo
- Short delays: 3w, 2w, 1w
- Very short delays: 3d, 2d, 1d

### Plotting Methods

#### Exponential Model (Parametric)

Fits exponential discounting model `V(t) = A * exp(-k*t)` with logit choice function:

```python
plot_discount_curves(data, method='exponential', output_file='exp_curves.png')
```

#### Switch Point Analysis (Nonparametric)

Calculates the delay where participants switch from preferring immediate to delayed rewards:

```python
plot_discount_curves(data, method='switch_point', output_file='switch_curves.png')
```

## API Reference

### `parse_japanese_delay(delay_str: str) -> float`

Parse Japanese delay string to numeric years.

**Parameters:**
- `delay_str`: Japanese delay string (e.g., "10年後")

**Returns:** Delay in years as float

**Raises:** `ValueError` if format is not recognized

### `load_discounting_data(filepath: str) -> pd.DataFrame`

Load delay discounting data from CSV file.

**Parameters:**
- `filepath`: Path to CSV file

**Returns:** DataFrame with parsed data including `t_years` column

### `fit_exponential_discounting(t_years, choices, immediate_amounts=None, delayed_amounts=None) -> dict`

Fit exponential discounting model with logit choice function.

**Parameters:**
- `t_years`: Array of delays in years
- `choices`: Binary array of choices (0 = immediate, 1 = delayed)
- `immediate_amounts`: Optional array of immediate amounts
- `delayed_amounts`: Optional array of delayed amounts

**Returns:** Dictionary with `'k'` (discount rate), `'beta'` (temperature), `'nll'` (negative log likelihood)

### `calculate_switch_point(t_years, choices) -> tuple`

Calculate switch point for nonparametric analysis.

**Parameters:**
- `t_years`: Array of delays in years
- `choices`: Binary array of choices

**Returns:** Tuple of `(switch_point_t, implied_discount_factor)`

### `plot_discount_curves(data, method='exponential', output_file=None, figsize=(10,6))`

Create overlaid plot of per-participant discount curves.

**Parameters:**
- `data`: DataFrame with columns: participant, t_years, choice
- `method`: `'exponential'` or `'switch_point'`
- `output_file`: Optional path to save figure
- `figsize`: Figure size tuple

**Returns:** Tuple of `(fig, ax)` matplotlib objects

### `generate_synthetic_data(n_participants=5, staircase_delays=None, output_file=None) -> pd.DataFrame`

Generate synthetic delay discounting data.

**Parameters:**
- `n_participants`: Number of participants to simulate
- `staircase_delays`: Optional list of delay strings
- `output_file`: Optional path to save CSV

**Returns:** DataFrame with synthetic data

## Testing

Run the unit tests:

```bash
python -m pytest test_delay_discounting.py -v
```

Or using unittest:

```bash
python -m unittest test_delay_discounting.py -v
```

## Examples

See `example.py` for a complete working example that demonstrates:
- Generating synthetic data
- Loading and parsing Japanese delay labels
- Fitting exponential models
- Calculating switch points
- Creating spaghetti plots

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.