def clean_data(df):
    df_clean = df.copy()
    df_clean["Sales"] = df_clean["Sales"].fillna(0)
    df_clean["Price"] = df_clean["Price"].ffill()
    df_clean["Promotion"] = df_clean["Promotion"].fillna(0)
    if "Current_Stock" not in df_clean.columns:
        df_clean["Current_Stock"] = 0
    df_clean = df_clean.drop_duplicates(subset=["SKU", "Date"])
    sales_upper_limit = df_clean["Sales"].quantile(0.99)
    df_clean["Sales"] = df_clean["Sales"].clip(upper=sales_upper_limit)
    df_clean = df_clean.sort_values(["SKU", "Date"]).reset_index(drop=True)
    return df_clean
