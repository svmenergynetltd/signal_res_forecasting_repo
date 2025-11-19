#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 11:32:30 2025

@author: balajivenkateswaran

"""

import re
import json
import warnings
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)

# Optional libraries
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None

# -------------------------- CONFIG --------------------------
class CONFIG:
    INPUT_FILE = "Data/Data_updated.xlsx"   # path to Excel input
    TIMESTAMP_COL = "Statistical Period"
    RESULTS_DIR = Path("results_json")
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    # Target options: a single column name or list -> will sum to combined target
    # Default: forecast PV Yield (kWh)
    TARGET_COLUMNS = ["PV Yield (kWh)"]  # e.g. ["PV Yield (kWh)"] or ["PV Yield (kWh)","Loss Due To Export Limitation (kWh)"]
    TARGET_NAME_FOR_FILES = "PV_Yield"   # used in filenames; set friendly name

    # Time & horizon
    H = 24  # hourly horizon (24 hours ahead)
    UPSAMPLE_TO_30MIN = True  # output in 30-min steps (True) or hourly (False)

    # Initial training cutoff - everything with index < INITIAL_CUTOFF is used for initial training
    INITIAL_CUTOFF = pd.to_datetime("2025-08-24 23:00:00")  # change as needed

    # Retraining frequency: 'daily', 'weekly', or None
    RETRAIN_FREQUENCY = 'daily'

    # Irradiation threshold for daytime metrics and zeroing rule
    IRR_DAY_THRESHOLD = 1e-4

    # Which models to try (set booleans)
    DO_LGB = True
    DO_XGB = True
    DO_RF = True
    DO_ET = True
    DO_GBR = True
    DO_LR = True
    DO_DT = True

    # Train hyperparameters (sane defaults)
    LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=-1)
    XGB_PARAMS = dict(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=4)
    RF_PARAMS = dict(n_estimators=200, random_state=42, n_jobs=-1)
    ET_PARAMS = dict(n_estimators=150, random_state=42, n_jobs=-1)
    GBR_PARAMS = dict(n_estimators=200, random_state=42)
    # Save trained model snapshots?
    SAVE_MODELS = False

# -------------------------- helpers --------------------------
def clean_colnames(df):
    return df.rename(columns={c: re.sub(r'\s+', ' ', str(c).strip()) for c in df.columns})

def parse_times(s):
    s = s.astype(str)
    s = s.str.replace(r'\s+[A-Za-z]{2,5}$', '', regex=True)
    s = s.str.replace(r'\s*\(.*\)$', '', regex=True)
    s = s.str.replace(r'UTC[+\-]\d{1,2}:?\d{0,2}', '', regex=True)
    return pd.to_datetime(s, errors='coerce')

def ensure_expected_columns(df, expected):
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan
    return df

def underscorize(cols):
    out = []
    for c in cols:
        s = str(c)
        s = re.sub(r'\s+', '_', s.strip())
        s = s.replace('(', '').replace(')', '').replace('/', '_per_').replace('°','deg').replace('㎡','m2')
        out.append(s)
    return out

def find_col(df, patterns):
    for pat in patterns:
        for c in df.columns:
            if pat.lower() in str(c).lower():
                return c
    return None

def upsample_hourly_to_30min_weighted(pred_hourly, irr_series, day_start):
    """Irradiation-weighted split of 24 hourly kWh -> 48 half-hours."""
    assert len(pred_hourly) == 24
    idx_30 = pd.date_range(start=day_start, periods=48, freq='30T')
    # construct hourly irradiation series and interpolate to 30-min
    irr_hour = irr_series.reindex(pd.date_range(irr_series.index.min(), irr_series.index.max(), freq='H'))
    irr_30 = irr_hour.reindex(idx_30).interpolate(method='time').ffill().bfill().fillna(0.0)
    vals_30 = []
    for h in range(24):
        t1 = day_start + pd.Timedelta(hours=h)
        t2 = t1 + pd.Timedelta(minutes=30)
        w1 = float(irr_30.get(t1, 0.0))
        w2 = float(irr_30.get(t2, 0.0))
        val = float(pred_hourly[h])
        if (w1 + w2) > 0:
            v1 = val * (w1 / (w1 + w2))
            v2 = val * (w2 / (w1 + w2))
        else:
            v1 = v2 = val / 2.0
        vals_30 += [v1, v2]
    return idx_30, np.array(vals_30)

def save_json_series(timestamps, values, outpath):
    arr = []
    for ts, v in zip(timestamps, values):
        val = None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        arr.append({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "data": val})
    with open(outpath, 'w') as f:
        json.dump(arr, f, indent=2)

# -------------------------- main pipeline --------------------------
def main():
    # Load
    if not Path(CONFIG.INPUT_FILE).exists():
        raise FileNotFoundError(f"Input file '{CONFIG.INPUT_FILE}' not found.")
    print("Loading input file...")
    df_raw = pd.read_excel(CONFIG.INPUT_FILE)
    df_raw = clean_colnames(df_raw)
    if CONFIG.TIMESTAMP_COL not in df_raw.columns:
        raise KeyError(f"Timestamp column '{CONFIG.TIMESTAMP_COL}' not found.")
    df_raw[CONFIG.TIMESTAMP_COL] = parse_times(df_raw[CONFIG.TIMESTAMP_COL])
    if df_raw[CONFIG.TIMESTAMP_COL].isna().any():
        nbad = df_raw[CONFIG.TIMESTAMP_COL].isna().sum()
        print(f"Warning: {nbad} timestamps could not be parsed and will be dropped.")
    df_raw = df_raw.dropna(subset=[CONFIG.TIMESTAMP_COL]).set_index(CONFIG.TIMESTAMP_COL).sort_index()

    # reindex hourly grid
    start = df_raw.index.min().floor('H')
    end = df_raw.index.max().ceil('H')
    df_hourly = df_raw.reindex(pd.date_range(start=start, end=end, freq='H'))
    # ensure expected cols
    expected = ['Global Irradiation (kWh/㎡)', 'Average Temperature(°C)', 'Theoretical Yield (kWh)',
                'PV Yield (kWh)', 'Inverter Yield (kWh)', 'Loss Due to Export Limitation (kWh)']
    df_hourly = ensure_expected_columns(df_hourly, expected)

    # impute pragmatically
    df_hourly['Loss Due to Export Limitation (kWh)'] = df_hourly['Loss Due to Export Limitation (kWh)'].fillna(0.0)
    for c in ['PV Yield (kWh)', 'Inverter Yield (kWh)', 'Theoretical Yield (kWh)']:
        if c in df_hourly.columns:
            df_hourly[c] = df_hourly[c].fillna(0.0)
    df_hourly['Average Temperature(°C)'] = pd.to_numeric(df_hourly['Average Temperature(°C)'], errors='coerce')
    df_hourly['Average Temperature(°C)'] = df_hourly['Average Temperature(°C)'].interpolate(method='time', limit_direction='both').ffill().bfill().fillna(20.0)
    df_hourly['Global Irradiation (kWh/㎡)'] = pd.to_numeric(df_hourly['Global Irradiation (kWh/㎡)'], errors='coerce')
    df_hourly['Global Irradiation (kWh/㎡)'] = df_hourly['Global Irradiation (kWh/㎡)'].interpolate(method='time', limit=6).fillna(0.0)

    # Prepare target: sum columns if list provided
    if isinstance(CONFIG.TARGET_COLUMNS, (list, tuple)) and len(CONFIG.TARGET_COLUMNS) > 1:
        for c in CONFIG.TARGET_COLUMNS:
            if c not in df_hourly.columns:
                df_hourly[c] = 0.0
        df_hourly['TARGET_COMBINED'] = df_hourly[CONFIG.TARGET_COLUMNS].fillna(0.0).sum(axis=1)
        target_col = 'TARGET_COMBINED'
    else:
        target_col = CONFIG.TARGET_COLUMNS[0]
        if target_col not in df_hourly.columns:
            raise KeyError(f"Requested target column '{target_col}' not found in the data.")

    print("Data hourly range:", df_hourly.index.min(), "->", df_hourly.index.max())
    print("Target column (raw):", target_col)

    # Feature engineering (hourly)
    df_feats = df_hourly.copy()
    df_feats['hour'] = df_feats.index.hour
    df_feats['dayofweek'] = df_feats.index.dayofweek
    df_feats['month'] = df_feats.index.month
    df_feats['is_weekend'] = df_feats['dayofweek'].isin([5,6]).astype(int)
    # lags and rolls
    for lag in [1, 24, 48]:
        df_feats[f'lag_{lag}'] = df_feats[target_col].shift(lag)
    df_feats['roll_3'] = df_feats[target_col].shift(1).rolling(3, min_periods=1).mean()
    df_feats['roll_6'] = df_feats[target_col].shift(1).rolling(6, min_periods=1).mean()
    df_feats['roll_24'] = df_feats[target_col].shift(1).rolling(24, min_periods=1).mean()
    df_feats = df_feats.dropna(subset=['lag_48'])  # require history

    # underscorize column names and pick features
    mapping = dict(zip(df_feats.columns, underscorize(df_feats.columns)))
    df_feats = df_feats.rename(columns=mapping)
    target_underscored = underscorize([target_col])[0]

    candidate_features = [
        mapping.get('Global Irradiation (kWh/㎡)', 'Global_Irradiation_kWh_m2'),
        mapping.get('Average Temperature(°C)', 'Average_TemperaturedegC'),
        mapping.get('Theoretical Yield (kWh)', 'Theoretical_Yield_kWh'),
        mapping.get('Inverter Yield (kWh)', 'Inverter_Yield_kWh'),
        mapping.get('Loss Due to Export Limitation (kWh)', 'Loss_Due_to_Export_Limitation_kWh'),
        'hour','dayofweek','month','is_weekend',
        'lag_1','lag_24','lag_48','roll_3','roll_6','roll_24'
    ]
    features = [f for f in candidate_features if f in df_feats.columns]
    print("Prepared multi-step features. Rows:", len(df_feats), "Features used:", features)

    # Build multi-output targets (y_t_plus_1 .. y_t_plus_24)
    H = CONFIG.H
    for h in range(1, H+1):
        df_feats[f'y_t_plus_{h}'] = df_feats[target_underscored].shift(-h)
    df_model = df_feats.dropna(subset=[f'y_t_plus_{H}']).copy()
    X_all = df_model[features].copy()
    Y_cols = [f'y_t_plus_{h}' for h in range(1, H+1)]
    Y_all = df_model[Y_cols].copy()
    print("Prepared multi-step dataset. Rows:", len(X_all), "Targets:", len(Y_cols))

    # Split initial training vs simulation (strict < to avoid leakage)
    train_mask = X_all.index < CONFIG.INITIAL_CUTOFF
    X_train = X_all.loc[train_mask].copy()
    Y_train = Y_all.loc[train_mask].copy()
    X_rest = X_all.loc[~train_mask].copy()
    Y_rest = Y_all.loc[~train_mask].copy()
    print(f"Initial training rows: {len(X_train)}. Rows remaining for simulation: {len(X_rest)}")

    # Train initial models
    models = {}
    # Helper to create MultiOutputRegressor from constructor
    def train_and_store(name, ctor):
        try:
            base = ctor()
            wrapped = MultiOutputRegressor(base, n_jobs=1)
            wrapped.fit(X_train, Y_train.values)
            models[name] = wrapped
            print(f" Trained: {name}")
        except Exception as e:
            print(f" Failed to train {name}: {e}")

    # list of model ctors depending on availability & config
    if CONFIG.DO_LR:
        pass  # placeholder to ensure attribute exists in config? we'll define DO_LR etc below

    # Train models conditionally
    if CONFIG.DO_LR:
        train_and_store('lr', lambda: LinearRegression())
    if CONFIG.DO_RF:
        train_and_store('rf', lambda: RandomForestRegressor(**CONFIG.RF_PARAMS))
    if CONFIG.DO_ET:
        train_and_store('et', lambda: ExtraTreesRegressor(**CONFIG.ET_PARAMS))
    if CONFIG.DO_GBR:
        train_and_store('gbr', lambda: GradientBoostingRegressor(**CONFIG.GBR_PARAMS))
    if CONFIG.DO_DT:
        train_and_store('dt', lambda: DecisionTreeRegressor(random_state=42))
    if CONFIG.DO_LGB and lgb is not None:
        train_and_store('lgb', lambda: lgb.LGBMRegressor(**CONFIG.LGB_PARAMS))
    elif CONFIG.DO_LGB:
        print("LightGBM not available; skipping lgb.")
    if CONFIG.DO_XGB and xgb is not None:
        train_and_store('xgb', lambda: xgb.XGBRegressor(**CONFIG.XGB_PARAMS))
    elif CONFIG.DO_XGB:
        print("XGBoost not available; skipping xgb.")

    print("Initial models trained:", list(models.keys()))

    # Forecast loop (daily by default)
    # Determine list of forecast days from the remainder range
    if len(X_rest) == 0:
        print("No data to simulate after initial cutoff.")
        return

    # Create daily list of days to forecast using the index of X_rest
    first_day = X_rest.index.min().floor('D')
    last_day = X_rest.index.max().floor('D')
    days = pd.date_range(start=first_day, end=last_day, freq='D')

    metrics = []
    cur_X = X_train.copy()
    cur_Y = Y_train.copy()

    for day in days:
        day0 = pd.Timestamp(day).normalize()
        origin_time = day0 - pd.Timedelta(hours=1)  # features timestamp used to predict day0
        # choose latest feature row <= origin_time
        feat_time = X_all.index[X_all.index <= origin_time].max()
        if pd.isna(feat_time):
            # nothing to predict for this day
            print(f"Skipping {day0.date()}: no available features before origin_time {origin_time}")
            continue
        feat_row = X_all.loc[[feat_time]]

        # For metrics we want the actual hourly target for the day (if available)
        hourly_idx = pd.date_range(day0, day0 + pd.Timedelta(hours=23), freq='H')
        actual_hourly = None
        if set(hourly_idx).issubset(set(df_hourly.index)):
            actual_hourly = df_hourly.loc[hourly_idx, target_col].values
        # irradiation series for upsampling
        irr_series = df_hourly['Global Irradiation (kWh/㎡)']

        # Predict with each model and save JSON 30-min
        for name, model in models.items():
            try:
                pred_hourly = np.maximum(model.predict(feat_row)[0], 0.0)  # length H
            except Exception as e:
                print(f" Prediction failed for {name} on {day0.date()}: {e}")
                continue

            # apply nighttime zero rule at hourly level before upsample:
            # if irrigation at the hour is zero -> set that hour's predicted energy to 0
            for i, t in enumerate(hourly_idx):
                irr_val = float(irr_series.get(t, 0.0))
                if irr_val <= CONFIG.IRR_DAY_THRESHOLD:
                    pred_hourly[i] = 0.0

            # upsample to 30-min
            if CONFIG.UPSAMPLE_TO_30MIN:
                ts30, vals30 = upsample_hourly_to_30min_weighted(pred_hourly, irr_series, day0)
            else:
                ts30 = hourly_idx
                vals30 = pred_hourly

            # Ensure non-negative
            vals30 = np.maximum(vals30, 0.0)

            # Save JSON
            fname = f"{day0.date()}_{name}_{CONFIG.TARGET_NAME_FOR_FILES}_{'30min' if CONFIG.UPSAMPLE_TO_30MIN else 'hourly'}.json"
            outpath = CONFIG.RESULTS_DIR / fname
            save_json_series(ts30, vals30, outpath)

            # compute simple MAE/RMSE if actuals available (upsample actuals to 30-min using same method)
            if actual_hourly is not None:
                if CONFIG.UPSAMPLE_TO_30MIN:
                    _, actual30 = upsample_hourly_to_30min_weighted(actual_hourly, irr_series, day0)
                else:
                    actual30 = actual_hourly
                mae_val = float(np.nanmean(np.abs(actual30 - vals30)))
                rmse_val = float(np.sqrt(np.nanmean((actual30 - vals30) ** 2)))
                # daytime mask
                irr_30 = irr_series.reindex(pd.date_range(day0, day0 + pd.Timedelta(hours=23, minutes=30), freq='30T')).interpolate(method='time').fillna(0.0)
                day_mask = irr_30.values > CONFIG.IRR_DAY_THRESHOLD
                day_mae = float(np.nanmean(np.abs(actual30[day_mask] - vals30[day_mask]))) if day_mask.sum() > 0 else None
            else:
                mae_val = None; rmse_val = None; day_mae = None

            metrics.append({'date': str(day0.date()), 'model': name, 'mae': mae_val, 'rmse': rmse_val, 'day_mae': day_mae, 'origin_time': str(feat_time)})
            print(f" Saved {outpath.name}  (model {name})  MAE: {mae_val}")

        # Append day rows to training sets for retrain (if available)
        mask_day = (X_all.index >= hourly_idx[0]) & (X_all.index <= hourly_idx[-1])
        if mask_day.sum() > 0:
            cur_X = pd.concat([cur_X, X_all.loc[mask_day]])
            cur_Y = pd.concat([cur_Y, Y_all.loc[mask_day]])

        # Retrain according to frequency
        if CONFIG.RETRAIN_FREQUENCY == 'daily':
            # retrain all models on cur_X, cur_Y
            print(" Retraining models (daily)...")
            for name in list(models.keys()):
                try:
                    if name == 'lgb' and lgb is not None:
                        base = lgb.LGBMRegressor(**CONFIG.LGB_PARAMS)
                    elif name == 'xgb' and xgb is not None:
                        base = xgb.XGBRegressor(**CONFIG.XGB_PARAMS)
                    elif name == 'rf':
                        base = RandomForestRegressor(**CONFIG.RF_PARAMS)
                    elif name == 'et':
                        base = ExtraTreesRegressor(**CONFIG.ET_PARAMS)
                    elif name == 'gbr':
                        base = GradientBoostingRegressor(**CONFIG.GBR_PARAMS)
                    elif name == 'lr':
                        base = LinearRegression()
                    elif name == 'dt':
                        base = DecisionTreeRegressor(random_state=42)
                    else:
                        print(f" Unknown model {name} for retrain; skipping.")
                        continue
                    wrapped = MultiOutputRegressor(base, n_jobs=1)
                    wrapped.fit(cur_X, cur_Y.values)
                    models[name] = wrapped
                    if CONFIG.SAVE_MODELS:
                        joblib.dump(wrapped, CONFIG.RESULTS_DIR / f"model_{name}_up_to_{cur_X.index.max().strftime('%Y%m%d%H')}.pkl")
                except Exception as e:
                    print(f" Retrain failed for {name}: {e}")

        elif CONFIG.RETRAIN_FREQUENCY == 'weekly':
            # retrain only at end of week (Sunday)
            if day0.weekday() == 6:
                print(" Retraining models (weekly)...")
                for name in list(models.keys()):
                    try:
                        if name == 'lgb' and lgb is not None:
                            base = lgb.LGBMRegressor(**CONFIG.LGB_PARAMS)
                        elif name == 'xgb' and xgb is not None:
                            base = xgb.XGBRegressor(**CONFIG.XGB_PARAMS)
                        elif name == 'rf':
                            base = RandomForestRegressor(**CONFIG.RF_PARAMS)
                        elif name == 'et':
                            base = ExtraTreesRegressor(**CONFIG.ET_PARAMS)
                        elif name == 'gbr':
                            base = GradientBoostingRegressor(**CONFIG.GBR_PARAMS)
                        elif name == 'lr':
                            base = LinearRegression()
                        elif name == 'dt':
                            base = DecisionTreeRegressor(random_state=42)
                        else:
                            continue
                        wrapped = MultiOutputRegressor(base, n_jobs=1)
                        wrapped.fit(cur_X, cur_Y.values)
                        models[name] = wrapped
                        if CONFIG.SAVE_MODELS:
                            joblib.dump(wrapped, CONFIG.RESULTS_DIR / f"model_{name}_up_to_{cur_X.index.max().strftime('%Y%m%d%H')}.pkl")
                    except Exception as e:
                        print(f" Weekly retrain failed for {name}: {e}")
        else:
            # no retrain
            pass

    # Save metrics dataframe
    df_metrics = pd.DataFrame(metrics)
    metrics_file = CONFIG.RESULTS_DIR / "metrics_summary.csv"
    df_metrics.to_csv(metrics_file, index=False)
    print("Saved metrics summary to:", metrics_file.resolve())

    # final best-model one-off forecast (pref order)
    pref = ['lgb', 'xgb', 'rf', 'et', 'gbr', 'lr', 'dt']
    best = next((p for p in pref if p in models), None)
    if best is not None:
        last_feat = X_all.iloc[[-1]]
        last_time = X_all.index.max()
        pred_hourly = np.maximum(models[best].predict(last_feat)[0], 0.0)
        for i, t in enumerate(pd.date_range((last_time + timedelta(hours=1)).floor('D'), periods=24, freq='H')):
            irr_val = float(df_hourly['Global Irradiation (kWh/㎡)'].get(t, 0.0))
            if irr_val <= CONFIG.IRR_DAY_THRESHOLD:
                pred_hourly[i] = 0.0
        if CONFIG.UPSAMPLE_TO_30MIN:
            ts30, vals30 = upsample_hourly_to_30min_weighted(pred_hourly, df_hourly['Global Irradiation (kWh/㎡)'], (last_time + timedelta(hours=1)).floor('D'))
        else:
            ts30 = pd.date_range((last_time + timedelta(hours=1)).floor('D'), periods=24, freq='H')
            vals30 = pred_hourly
        save_json_series(ts30, vals30, CONFIG.RESULTS_DIR / "next_day_forecast_final.json")
        print("Final next-day forecast saved to:", (CONFIG.RESULTS_DIR / "next_day_forecast_final.json").resolve())

    print("All done. JSON outputs saved to:", CONFIG.RESULTS_DIR.resolve())

if __name__ == "__main__":
    main()
