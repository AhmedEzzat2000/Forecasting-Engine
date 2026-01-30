import pandas as pd
import joblib
from .data_ingestion import load_and_map_excel
from .data_cleaning import clean_data
from .feature_engineering import feature_engineering
from .reorder_optimization import calculate_safety_stock

def forecast_and_reorder(sme_excel_file, feature_columns, global_model_file="global_demand_model.pkl",
                         forecast_days=30, lead_time_days=7, service_level_z=1.65):
    df_sme = load_and_map_excel(sme_excel_file)
    df_sme = clean_data(df_sme)
    df_sme = feature_engineering(df_sme)
    model = joblib.load(global_model_file)
    X_sme = df_sme[feature_columns]
    y_sme = df_sme["Sales"]
    model.fit(X_sme, y_sme)
    safety_stock_dict = calculate_safety_stock(df_sme, lead_time_days=lead_time_days, service_level_z=service_level_z)
    forecast_list = []
    for sku in df_sme["SKU"].unique():
        last_row = df_sme[df_sme["SKU"]==sku].tail(1)
        X_input = last_row[feature_columns].copy()
        for day in range(forecast_days):
            predicted_sales = model.predict(X_input)[0]
            forecast_list.append({"SKU": sku, "Day": day+1, "Forecast_Sales": predicted_sales})
            for lag_days in [1,7,30]:
                col_name = f"Sales_Lag_{lag_days}"
                if col_name in X_input.columns:
                    X_input[col_name] = predicted_sales
    forecast_df = pd.DataFrame(forecast_list)
    forecast_df["Safety_Stock"] = forecast_df["SKU"].map(safety_stock_dict)
    forecast_df["Reorder_Qty"] = (forecast_df["Forecast_Sales"] + forecast_df["Safety_Stock"]).astype(int)
    output_file = "SME_forecast_reorder.xlsx"
    forecast_df.to_excel(output_file, index=False)
    print(f"Forecast & Reorder Excel saved: {output_file}")
    return forecast_df
