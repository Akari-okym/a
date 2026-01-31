# Discount Curve Estimation

A Python tool for estimating and visualizing discount curves from intertemporal choice data.

## Features

- **Binary Choice Encoding**: Uses 0 for immediate choice, 1 for delayed choice
- **Day-based Delays**: All time delays are standardized to integer days
- **Multi-language Support**: Parses both Japanese (年/月/週/日) and English (y/mo/w/d) delay labels
- **Staircase Delays**: Handles standard staircase delay sequences (10y, 7y, 5y, 3y, 1y, 6mo, 2mo, 1mo, 3w, 2w, 1w, 3d, 2d, 1d)
- **Multiple Models**: Supports hyperbolic and exponential discount functions
- **Visualization**: Generates plots with delay in days on x-axis

## Installation

Requires Python 3.6+ with the following packages:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install pandas numpy matplotlib scipy
```

## Usage

### Command Line

Run with an example CSV file:

```bash
python discount_curve.py example_choices.csv
```

Run with custom model (hyperbolic or exponential):

```bash
python discount_curve.py example_choices.csv hyperbolic
python discount_curve.py example_choices.csv exponential
```

Run without arguments to generate example data:

```bash
python discount_curve.py
```

### CSV Format

The input CSV file should have the following columns:

- `delay`: Time delay (e.g., "10y", "6mo", "3w", "1d", "1年", "6月", "3週", "1日")
- `choice`: Binary choice (0 = immediate, 1 = delayed)

Example (`example_choices.csv`):

```csv
delay,choice
10y,0
7y,0
5y,0
3y,1
1y,1
6mo,1
2mo,1
1mo,1
3w,1
2w,1
1w,1
3d,1
2d,1
1d,1
```

### Python API

```python
from discount_curve import (
    load_choice_data,
    estimate_discount_rate,
    plot_discount_curve,
    parse_delay_to_days
)

# Load data
df = load_choice_data('example_choices.csv')

# Estimate discount rate (returns k in units of per day)
k = estimate_discount_rate(df, model='hyperbolic')
print(f"Discount rate: {k:.6f} per day")
print(f"Annual rate: {k * 365:.4f} per year")

# Plot the curve
plot_discount_curve(df, k, model='hyperbolic', save_path='curve.png')

# Parse individual delays
days = parse_delay_to_days("10y")   # Returns 3650
days = parse_delay_to_days("6mo")   # Returns 180
days = parse_delay_to_days("3w")    # Returns 21
days = parse_delay_to_days("1d")    # Returns 1
days = parse_delay_to_days("1年")   # Returns 365 (Japanese)
days = parse_delay_to_days("6月")   # Returns 180 (Japanese)
```

## Units and Conversions

All time delays are converted to **integer days**:

- **Year (y, 年)**: 365 days
- **Month (mo, 月)**: 30 days
- **Week (w, 週)**: 7 days
- **Day (d, 日)**: 1 day

The discount rate parameter `k` is estimated in units of **per day**. To convert to annual rate:
```python
k_annual = k_per_day * 365
```

## Models

### Hyperbolic Discounting

Discount function: `V(t) = 1 / (1 + k*t)`

Where:
- `t` is time delay in days (integer)
- `k` is the discount rate parameter (per day)

### Exponential Discounting

Discount function: `V(t) = exp(-k*t)`

Where:
- `t` is time delay in days (integer)
- `k` is the discount rate parameter (per day)

## Output

The script generates:

1. **Console output**: Estimated discount rate in per-day and per-year units
2. **Plot file** (`discount_curve.png`): Visualization showing:
   - Fitted discount curve with delay in days on x-axis
   - Observed choices (green circles = delayed, red X's = immediate)
   - Discount factor (0-1) on y-axis

## Standard Staircase Delays

The following standard delay sequence is commonly used in intertemporal choice experiments:

```python
['10y', '7y', '5y', '3y', '1y', '6mo', '2mo', '1mo', '3w', '2w', '1w', '3d', '2d', '1d']
```

In days:
```
10y  = 3650 days
7y   = 2555 days
5y   = 1825 days
3y   = 1095 days
1y   = 365 days
6mo  = 180 days
2mo  = 60 days
1mo  = 30 days
3w   = 21 days
2w   = 14 days
1w   = 7 days
3d   = 3 days
2d   = 2 days
1d   = 1 day
```

## License

MIT