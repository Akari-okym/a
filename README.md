# Discount Curve Visualization

Python tools for estimating and visualizing individual discount functions from binary choice survey data.

## Overview

This project provides tools to analyze temporal discounting behavior from binary choice experiments where participants choose between:
- **Immediate option**: Receive 10,000 JPY now
- **Delayed option**: Receive 11,000 JPY after delay `t`

The analysis estimates each participant's discount rate using exponential discounting with a probabilistic (logit) choice rule, then creates "spaghetti plots" showing all participants' discount curves overlaid.

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0

## Quick Start

### 1. Prepare Your Data

Create a CSV file with the following columns:

| Column | Description | Type | Example Values |
|--------|-------------|------|----------------|
| `participant_id` | Unique participant identifier | string/int | "P001", "P002", 1, 2 |
| `t` | Time delay (specify units consistently) | numeric | 7, 14, 30, 60, 90, 180 |
| `choice` | Binary choice indicator | binary/string | 1/0, "delayed"/"immediate" |

**Example CSV (`data/choices.csv`):**
```csv
participant_id,t,choice
P001,7,delayed
P001,14,delayed
P001,30,immediate
P001,60,immediate
P002,7,delayed
P002,14,immediate
...
```

**Supported choice encodings:**
- Binary: `0` (immediate), `1` (delayed)
- Text: `"immediate"`/`"delayed"`, `"now"`/`"later"`, `"a"`/`"b"`, `"A"`/`"B"`
- Both uppercase and lowercase are supported

### 2. Generate Sample Data (Optional)

To test the tools with synthetic data:

```bash
cd scripts
python generate_sample_data.py
```

This creates `data/sample_choices.csv` with 20 synthetic participants.

### 3. Run the Analysis

Basic usage:

```bash
cd scripts
python plot_discount_curves.py ../data/sample_choices.csv
```

This will:
1. Load and validate the data
2. Estimate discount rate for each participant using maximum likelihood
3. Create a spaghetti plot showing all discount curves
4. Save the plot as `discount_curves.png`

**With options:**

```bash
python plot_discount_curves.py ../data/sample_choices.csv \
    --output my_plot.png \
    --save-estimates estimates.csv \
    --plot-type value \
    --delay-unit days \
    --show
```

### 4. Command-Line Options

```
positional arguments:
  input_csv             Path to input CSV file

optional arguments:
  --output, -o          Output file path for plot (default: discount_curves.png)
  --participant-col     Name of participant ID column (default: participant_id)
  --delay-col           Name of delay column (default: t)
  --choice-col          Name of choice column (default: choice)
  --amount-delayed      Amount of delayed reward (default: 11000)
  --amount-immediate    Amount of immediate reward (default: 10000)
  --plot-type           Plot 'value' (discounted JPY) or 'factor' (0-1) (default: value)
  --delay-unit          Unit of delay for axis label (default: days)
  --show                Show plot interactively
  --save-estimates      Save estimated parameters to CSV file
```

## Output

### 1. Discount Curves Plot

The script generates a "spaghetti plot" showing:
- **Individual curves** (light blue, transparent): Each participant's estimated discount curve
- **Median curve** (dark red, bold): The median discount rate across all participants
- **Immediate option** (black dashed line): Horizontal reference line at 10,000 JPY

**Two plot types:**
- `--plot-type value`: Y-axis shows discounted value in JPY (default)
  - Curves show `V(t) = 11000 × exp(-r×t)`
  - Easier to interpret in terms of actual choices
- `--plot-type factor`: Y-axis shows discount factor (0-1)
  - Curves show `D(t) = exp(-r×t)`
  - Shows pure discounting independent of amounts

### 2. Console Output

The script prints:
- Data validation results
- Number of participants and observations
- Summary statistics of estimated discount rates (median, mean, SD, min, max)
- Convergence information
- Edge case warnings (participants who always chose immediate/delayed)

### 3. Optional Estimates CSV

Use `--save-estimates output.csv` to save a CSV with:

| Column | Description |
|--------|-------------|
| `participant_id` | Participant identifier |
| `r` | Estimated discount rate parameter |
| `convergence` | Whether optimization converged (True/False) |
| `n_obs` | Number of observations for this participant |
| `prop_delayed` | Proportion of delayed choices |
| `edge_case` | Edge case type (None, 'always_immediate', 'always_delayed') |

## Model Details

### Exponential Discounting

Each participant's discount function is modeled as:

```
D_i(t) = exp(-r_i × t)
```

where:
- `D_i(t)` is participant `i`'s discount factor at delay `t`
- `r_i` is participant `i`'s discount rate (higher = more impatient)
- `t` is the time delay

### Probabilistic Choice (Logit Rule)

The probability of choosing the delayed option is:

```
P(choose delayed) = 1 / (1 + exp(-(V_delayed - V_immediate)))
```

where:
- `V_delayed = 11000 × D_i(t)`
- `V_immediate = 10000`

### Parameter Estimation

For each participant, we estimate `r_i` by maximizing the log-likelihood:

```
LL = Σ [choice_t × log(P_delayed) + (1-choice_t) × log(1-P_delayed)]
```

Optimization uses `scipy.optimize.minimize_scalar` with bounded search (`r` ∈ [1e-6, 10.0]).

### Edge Cases

**Always choose immediate:** Set `r = 10.0` (upper bound) with warning
**Always choose delayed:** Set `r = 1e-6` (lower bound) with warning

These participants provide limited information about their discount rate, so we assign boundary values.

## Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/                             # Data directory
│   └── sample_choices.csv            # Example synthetic data
├── scripts/                          # Main scripts
│   ├── discount_curve_utils.py       # Core estimation functions
│   ├── plot_discount_curves.py       # Main plotting script
│   └── generate_sample_data.py       # Generate synthetic test data
└── tests/                            # Tests
    └── test_discount_curves.py       # Unit tests
```

## Testing

Run the test suite:

```bash
pip install pytest  # if not already installed
pytest tests/test_discount_curves.py -v
```

Tests cover:
- Discount function calculations
- Probability calculations
- Parameter estimation (normal and edge cases)
- Data validation
- Multi-participant analysis

## Interpretation Guide

### Discount Rate (`r`)

- **Low r (< 0.01)**: Patient, values future highly
- **Medium r (0.01-0.05)**: Typical range for delays in days/weeks
- **High r (> 0.05)**: Impatient, strong preference for immediate

### Discount Factor (`D(t)`)

At delay `t`, the delayed option is valued at:
```
V(t) = 11000 × D(t)
```

When `V(t) > 10000`, participant likely chooses delayed.
When `V(t) < 10000`, participant likely chooses immediate.

### Spaghetti Plot Interpretation

- **Wide spread**: High individual differences in patience
- **Steep curves**: Rapid discounting, prefer immediate rewards
- **Flat curves**: Slow discounting, patient
- **Curves crossing 10,000**: Indifference point (50% choose delayed)

## Assumptions and Limitations

1. **Exponential discounting**: May not capture all discounting patterns (e.g., hyperbolic)
2. **Fixed amounts**: Only works with one immediate/delayed amount pair
3. **Independent observations**: Assumes choices at different delays are independent
4. **No measurement error**: Assumes choices reflect true preferences
5. **Delay units**: Must be consistent across all observations (e.g., all in days)

## Troubleshooting

**"Missing required columns"**
- Ensure CSV has `participant_id`, `t`, and `choice` columns
- Use `--participant-col`, `--delay-col`, `--choice-col` if names differ

**"Choice column must contain only 0 or 1"**
- Check choice encoding matches supported formats
- Use binary (0/1) or text ("immediate"/"delayed")

**Many edge case warnings**
- Some participants always chose one option
- This is expected with extreme discounting or limited delay range
- Consider excluding these participants or expanding delay range

**Low convergence rate**
- May indicate insufficient data per participant
- Ensure each participant has multiple observations across different delays

## Citation

If you use this code in research, please cite appropriately and describe the exponential discount model with logit choice rule used for estimation.

## License

[Add your license here]

## Contact

[Add contact information here]