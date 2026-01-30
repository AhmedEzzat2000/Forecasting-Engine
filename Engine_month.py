import pandas as pd
import numpy as np
import joblib
from fuzzywuzzy import process
from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping, log_evaluation

STANDARD_COLUMNS = ["SKU", "Date", "Sales", "Price", "Promotion", "Current_Stock"]

def load_and_map_excel(file_path):
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

def clean_data(df):
    df_clean = df.copy()
    df_clean["Sales"] = df_clean["Sales"].fillna(0)
    df_clean["Price"] = df_clean["Price"].ffill()
    df_clean["Promotion"] = df_clean["Promotion"].fillna(0)
    if "Current_Stock" not in df_clean.columns:
        df_clean["Current_Stock"] = 0
    df_clean = df_clean.drop_duplicates(subset=["SKU","Date"])
    df_clean["Sales"] = df_clean["Sales"].clip(upper=df_clean["Sales"].quantile(0.99))
    df_clean = df_clean.sort_values(["SKU","Date"]).reset_index(drop=True)
    return df_clean

def feature_engineering(df):
    df_feat = df.copy()
    df_feat["DayOfWeek"] = df_feat["Date"].dt.dayofweek
    df_feat["Month"] = df_feat["Date"].dt.month
    df_feat["Quarter"] = df_feat["Date"].dt.quarter
    df_feat["IsWeekend"] = df_feat["DayOfWeek"].isin([5,6]).astype(int)
    df_feat = df_feat.sort_values(["SKU","Date"])
    for lag in [1,7,30]:
        df_feat[f"Sales_Lag_{lag}"] = df_feat.groupby("SKU")["Sales"].shift(lag)
    for window in [7,14,30]:
        df_feat[f"Sales_RollMean_{window}"] = df_feat.groupby("SKU")["Sales"].shift(1).rolling(window).mean()
        df_feat[f"Sales_RollStd_{window}"] = df_feat.groupby("SKU")["Sales"].shift(1).rolling(window).std()
    df_feat["Discount_pct"] = (df_feat["Price"].shift(1) - df_feat["Price"]) / df_feat["Price"].shift(1)
    df_feat["Discount_pct"] = df_feat["Discount_pct"].fillna(0)
    df_feat = df_feat.dropna()
    return df_feat

def train_global_model(df_feat, model_file="global_demand_model.pkl"):
    feature_columns = [c for c in df_feat.columns if c not in ["SKU","Date","Sales"]]
    df_feat = df_feat.sort_values(["SKU","Date"]).reset_index(drop=True)
    split_index = int(len(df_feat) * 0.8)
    train_df = df_feat.iloc[:split_index]
    val_df = df_feat.iloc[split_index:]
    X_train = train_df[feature_columns]
    y_train = train_df["Sales"]
    X_val = val_df[feature_columns]
    y_val = val_df["Sales"]
    model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31)
    callbacks_list = [early_stopping(stopping_rounds=50, verbose=True), log_evaluation(period=10)]
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks_list)
    joblib.dump(model, model_file)
    return model, feature_columns

def calculate_safety_stock(df, sales_col="Sales", lead_time_days=30, service_level_z=1.65):
    safety_stock = {}
    for sku, sku_group in df.groupby("SKU"):
        daily_std = sku_group[sales_col].std()
        safety_stock[sku] = int(np.ceil(service_level_z * daily_std * np.sqrt(lead_time_days)))
    return safety_stock

def forecast_next_month(sme_excel_file, model_file="global_demand_model.pkl", forecast_days=30,
                        lead_time_days=30, service_level_z=1.65):
    df_sme = load_and_map_excel(sme_excel_file)
    df_sme = clean_data(df_sme)
    df_sme = feature_engineering(df_sme)
    model = joblib.load(model_file)
    X_sme = df_sme[[c for c in df_sme.columns if c not in ["SKU","Date","Sales"]]]
    y_sme = df_sme["Sales"]
    model.fit(X_sme, y_sme)

    safety_stock_dict = calculate_safety_stock(df_sme, lead_time_days=lead_time_days, service_level_z=service_level_z)

    forecast_list = []
    for sku in df_sme["SKU"].unique():
        last_row = df_sme[df_sme["SKU"]==sku].tail(1)
        X_input = last_row[[c for c in df_sme.columns if c not in ["SKU","Date","Sales"]]].copy()
        predicted_daily_sales = model.predict(X_input)[0]
        predicted_month_sales = predicted_daily_sales * forecast_days
        forecast_list.append({
            "SKU": sku,
            "Forecast_Month_Sales": predicted_month_sales,
            "Safety_Stock": safety_stock_dict[sku],
            "Reorder_Qty": int(predicted_month_sales + safety_stock_dict[sku])
        })

    forecast_df = pd.DataFrame(forecast_list)
    forecast_df.to_excel("SME_forecast_next_month.xlsx", index=False)
    print("Forecast & Reorder Excel saved: SME_forecast_next_month.xlsx")
    return forecast_df

if __name__ == "__main__":
    df_sme = load_and_map_excel("sme_data.xlsx")
    df_sme = clean_data(df_sme)
    df_sme = feature_engineering(df_sme)
    global_model, feature_cols = train_global_model(df_sme)
    forecast_df = forecast_next_month("sme_data.xlsx", forecast_days=30)
