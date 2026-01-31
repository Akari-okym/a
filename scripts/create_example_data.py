"""
Create a more realistic example dataset with manually crafted choice patterns.
"""

import pandas as pd
import numpy as np

def create_example_data():
    """
    Create example data with diverse, manually specified choice patterns.
    """
    np.random.seed(42)
    
    delays = [7, 14, 30, 60, 90, 180]
    data = []
    
    # Very patient participants (mostly choose delayed)
    for i in range(1, 6):
        for t in delays:
            # Mostly delayed, some noise
            if t < 180:
                choice = 1 if np.random.random() > 0.1 else 0
            else:
                choice = 1 if np.random.random() > 0.3 else 0
            data.append({'participant_id': f'P{i:03d}', 't': t, 'choice': choice})
    
    # Moderately patient (switch around 60-90 days)
    for i in range(6, 16):
        for t in delays:
            if t < 30:
                choice = 1 if np.random.random() > 0.1 else 0
            elif t < 90:
                choice = 1 if np.random.random() > 0.5 else 0
            else:
                choice = 1 if np.random.random() > 0.8 else 0
            data.append({'participant_id': f'M{i-5:03d}', 't': t, 'choice': choice})
    
    # Less patient (switch around 14-30 days)
    for i in range(16, 26):
        for t in delays:
            if t <= 14:
                choice = 1 if np.random.random() > 0.2 else 0
            elif t <= 30:
                choice = 1 if np.random.random() > 0.6 else 0
            else:
                choice = 1 if np.random.random() > 0.9 else 0
            data.append({'participant_id': f'I{i-15:03d}', 't': t, 'choice': choice})
    
    # Very impatient (mostly choose immediate)
    for i in range(26, 31):
        for t in delays:
            if t == 7:
                choice = 1 if np.random.random() > 0.4 else 0
            else:
                choice = 1 if np.random.random() > 0.9 else 0
            data.append({'participant_id': f'V{i-25:03d}', 't': t, 'choice': choice})
    
    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    df = create_example_data()
    
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, 'sample_choices.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} observations for {df['participant_id'].nunique()} participants")
    print(f"Saved to: {output_path}")
    
    print("\nFirst few rows:")
    print(df.head(10))
    
    print("\nChoice distribution by delay:")
    print(df.groupby('t')['choice'].agg(['mean', 'count']))
    
    print("\nChoice proportion by participant group:")
    df['group'] = df['participant_id'].str[0]
    print(df.groupby('group')['choice'].agg(['mean', 'count']))
