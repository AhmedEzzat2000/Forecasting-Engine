import numpy as np

def calculate_safety_stock(df, sales_col="Sales", lead_time_days=7, service_level_z=1.65):
    safety_stock = {}
    for sku, sku_group in df.groupby("SKU"):
        daily_std = sku_group[sales_col].std()
        safety_stock[sku] = int(np.ceil(service_level_z * daily_std * np.sqrt(lead_time_days)))
    return safety_stock
