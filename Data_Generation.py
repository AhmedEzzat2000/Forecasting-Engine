import pandas as pd
import numpy as np
from datetime import datetime
import random

def generate_dummy_qatari_sme_data(
    num_skus=10,
    start_date="2024-01-01",
    end_date="2025-12-31"
):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start, end, freq="D")

    all_data = []

    ramadan_months = [3, 4]
    qatar_national_day = (12, 18)

    for sku_id in range(1, num_skus + 1):
        base_price = round(random.uniform(15, 150), 2)

        month_multiplier = {
            1: 0.9, 2: 0.85, 3: 1.0, 4: 1.05, 5: 1.1, 6: 1.2,
            7: 1.15, 8: 1.05, 9: 0.95, 10: 1.0, 11: 1.15, 12: 1.3
        }

        sales, price, promo, stock = [], [], [], []
        ramadan_flag, holiday_flag = [], []
        temperature, foot_traffic, competitor_price = [], [], []

        for date in dates:
            dow = date.weekday()
            season = month_multiplier[date.month]

            is_ramadan = 1 if date.month in ramadan_months else 0
            is_holiday = 1 if (date.month, date.day) == qatar_national_day else 0

            base_sales = np.random.poisson(lam=25)

            if dow >= 5:
                base_sales *= 1.1

            if is_ramadan:
                base_sales *= 1.15

            if is_holiday:
                base_sales *= 1.4

            is_promo = np.random.binomial(1, 0.12)

            daily_sales = int(
                base_sales * season * (1.25 if is_promo else 1)
            )

            daily_price = round(
                base_price * (1 - 0.07 * is_promo) * np.random.uniform(0.95, 1.05),
                2
            )

            temp = round(
                22 + 15 * np.sin((date.month - 1) / 12 * 2 * np.pi) + np.random.normal(0, 2),
                1
            )

            traffic = int(
                np.clip(np.random.normal(70 + 10 * is_holiday, 8), 40, 120)
            )

            competitor_idx = round(
                np.random.uniform(0.9, 1.1),
                2
            )

            current_stock = max(daily_sales + np.random.randint(5, 20), 0)

            sales.append(daily_sales)
            price.append(daily_price)
            promo.append(is_promo)
            stock.append(current_stock)
            ramadan_flag.append(is_ramadan)
            holiday_flag.append(is_holiday)
            temperature.append(temp)
            foot_traffic.append(traffic)
            competitor_price.append(competitor_idx)

        df_sku = pd.DataFrame({
            "SKU": f"SKU_{sku_id}",
            "Date": dates,
            "Sales": sales,
            "Price": price,
            "Promotion": promo,
            "Current_Stock": stock,
            "Is_Ramadan": ramadan_flag,
            "Is_Holiday": holiday_flag,
            "Temperature": temperature,
            "Foot_Traffic_Index": foot_traffic,
            "Competitor_Price_Index": competitor_price
        })

        all_data.append(df_sku)

    return pd.concat(all_data, ignore_index=True)


sme_data = generate_dummy_qatari_sme_data(num_skus=1000)
sme_data.to_excel("sme_qatar_data.xlsx", index=False)
print("Qatari SME dummy data generated successfully")
