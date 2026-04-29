import pandas as pd
import numpy as np
from pathlib import Path

def get_most_frequent_value(series):
    """Returns the most frequent value ignoring nulls. If empty, returns an empty string."""
    mode = series.dropna().mode()
    return mode.iloc[0] if not mode.empty else ""

def calculate_abnormal_percentage(series):
    """Calculates the percentage of 'abnormal' flags in the group."""
    total = len(series.dropna())
    if total == 0:
        return 0.0
    # In MIMIC the anomaly flag is typically the string 'abnormal'
    abnormal_count = (series.str.lower() == 'abnormal').sum()
    return round((abnormal_count / total) * 100, 2)

def calculate_cluster_statistics(df_lab):
    """
    Groups lab data and calculates statistics for each profile.
    """
    print("⏳ Starting grouping and statistics calculation...")
    
    # Define aggregation rules for each column
    aggregations = {
        # Numeric statistics (pandas automatically ignores NaN in min/max/mean/median)
        'valuenum': ['min', 'max', 'mean', 'median'],
        
        # Categorical values and notes
        'value': [('most_frequent_value', get_most_frequent_value)],
        
        # Count how many tests are in this group (useful for discarding small groups)
        'valueuom': [('sample_count', 'count')]
    }
    
    grouping_columns = [
        'value',
        'valuenum',
        'valueuom',
        'ref_range_lower',
        'ref_range_upper',
    ]

    # Execute grouping
    df_grouped = df_lab.groupby(grouping_columns).agg(aggregations)
    
    # --- Output formatting ---
    # The agg() operation creates a MultiIndex on columns (e.g., ('valuenum', 'min')).
    # We flatten it to have simple and clean column names.
    df_grouped.columns = [
        f"{col[0]}_{col[1]}" if col[1] not in ['most_frequent_value', 'most_frequent_comment', 'abnormal_pct', 'sample_count'] 
        else col[1] 
        for col in df_grouped.columns
    ]
    
    # Transform grouping indices into normal columns
    df_grouped = df_grouped.reset_index()
    
    # Sort by number of samples (from largest to smallest group)
    df_grouped = df_grouped.sort_values(by='sample_count', ascending=False).reset_index(drop=True)
    
    print(f"✅ Calculation completed. Created {len(df_grouped)} unique clinical clusters.")
    
    return df_grouped

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    input_csv = Path("datasets/labevents_cleaned.csv")
    input_csv_typo = Path("datasets/labevents_cleaned.csv")  # fallback typo
    output_csv = Path("datasets/clinical_clusters.csv")

    # fallback if file has typo in name
    if not input_csv.exists() and input_csv_typo.exists():
        input_csv = input_csv_typo

    try:
        print(f"⏳ Loading input from: {input_csv}")
        df_lab = pd.read_csv(input_csv)

        df_statistics = calculate_cluster_statistics(df_lab)

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_statistics.to_csv(output_csv, index=False)
        print(f"\n💾 CSV saved to: {output_csv}")

        print("\nResult preview:")
        pd.set_option('display.max_columns', None)
        print(df_statistics.head())

    except FileNotFoundError:
        print("❌ Error: input file not found in datasets/.")
        print("   Expected: datasets/labevents_cleaned.csv")
        print("   Or: datasets/labevents_cleaned.csv")