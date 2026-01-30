def feature_engineering(df):
    df_feat = df.copy()
    df_feat["DayOfWeek"] = df_feat["Date"].dt.dayofweek
    df_feat["Month"] = df_feat["Date"].dt.month
    df_feat["Quarter"] = df_feat["Date"].dt.quarter
    df_feat["IsWeekend"] = df_feat["DayOfWeek"].isin([5,6]).astype(int)
    df_feat = df_feat.sort_values(["SKU", "Date"])
    for lag_days in [1,7,30]:
        df_feat[f"Sales_Lag_{lag_days}"] = df_feat.groupby("SKU")["Sales"].shift(lag_days)
    for window_days in [7,14,30]:
        df_feat[f"Sales_RollMean_{window_days}"] = df_feat.groupby("SKU")["Sales"].shift(1).rolling(window_days).mean()
        df_feat[f"Sales_RollStd_{window_days}"] = df_feat.groupby("SKU")["Sales"].shift(1).rolling(window_days).std()
    df_feat["Discount_pct"] = (df_feat["Price"].shift(1) - df_feat["Price"]) / df_feat["Price"].shift(1)
    df_feat["Discount_pct"] = df_feat["Discount_pct"].fillna(0)
    df_feat = df_feat.dropna()
    return df_feat
