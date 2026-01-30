import pandas as pd
from fuzzywuzzy import process

STANDARD_COLUMNS = ["SKU", "Date", "Sales", "Price", "Promotion", "Current_Stock"]

def load_and_map_excel(file_path: str) -> pd.DataFrame:
    raw_df = pd.read_excel(file_path)
    mapped_cols = {}
    for col in raw_df.columns:
        best_match, score = process.extractOne(col, STANDARD_COLUMNS)
        if score >= 80:
            mapped_cols[col] = best_match
    df_mapped = raw_df.rename(columns=mapped_cols)
    df_mapped = df_mapped[[c for c in STANDARD_COLUMNS if c in df_mapped.columns]]
    df_mapped["Date"] = pd.to_datetime(df_mapped["Date"])
    return df_mapped
