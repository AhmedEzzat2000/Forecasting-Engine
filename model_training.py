import joblib
from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping, log_evaluation

def train_global_model(df_feat, model_file="global_demand_model.pkl"):
    feature_columns = [c for c in df_feat.columns if c not in ["SKU", "Date", "Sales"]]
    df_feat = df_feat.sort_values(["SKU", "Date"]).reset_index(drop=True)
    split_index = int(len(df_feat) * 0.8)
    train_df = df_feat.iloc[:split_index]
    val_df = df_feat.iloc[split_index:]
    X_train = train_df[feature_columns]
    y_train = train_df["Sales"]
    X_val = val_df[feature_columns]
    y_val = val_df["Sales"]

    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31
    )

    callbacks_list = [
        early_stopping(stopping_rounds=50, verbose=True),
        log_evaluation(period=10)
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        #callbacks=callbacks_list
    )

    joblib.dump(model, model_file)
    return model, feature_columns
