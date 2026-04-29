import pandas as pd
from pathlib import Path


def extract_and_clean_labevents(file_path: Path) -> pd.DataFrame:
    """
    Extract relevant columns from labevents.csv.gz and apply basic cleaning
    to prepare for semantic grouping.
    """
    target_columns = [
        "value",
        "valuenum",
        "valueuom",
        "ref_range_lower",
        "ref_range_upper",
    ]

    print("⏳ Loading dataset... (this may require significant RAM)")

    # Explicit gzip read
    df = pd.read_csv(file_path, usecols=target_columns, compression="gzip")

    print(f"✅ File loaded. Total original rows: {len(df)}")

    # Basic cleaning - Exclude rows with null in any of: value, valuenum, valueuom
    df = df.dropna(subset=["value", "valuenum", "valueuom"], how="any")
    df["ref_range_lower"] = df["ref_range_lower"].fillna("NO_LOWER")
    df["ref_range_upper"] = df["ref_range_upper"].fillna("NO_UPPER")

    print(f"🧹 Cleaning completed. Remaining usable rows: {len(df)}")
    return df


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    input_gz_path = Path("datasets/mimiciv_demo/hosp/labevents.csv.gz")
    output_csv_path = Path("datasets/labevents_cleaned.csv")

    try:
        df_lab = extract_and_clean_labevents(input_gz_path)

        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_lab.to_csv(output_csv_path, index=False)

        print(f"\n💾 Cleaned file saved to: {output_csv_path}")
        print("\nPreview of extracted data:")
        print(df_lab.head())

    except FileNotFoundError:
        print(f"❌ Error: File not found at path {input_gz_path}")