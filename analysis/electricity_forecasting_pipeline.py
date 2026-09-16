#!/usr/bin/env python3
"""
End-to-end household electricity forecasting pipeline.

Pipeline:
1) Load raw minute-level household power CSV.
2) Restore date/datetime from a configurable start date.
3) Fill missing gaps:
   - <= 5 min: linear interpolation
   - 6..60 min: linear interpolation + local median fallback
   - > 60 min: seasonal median at same minute-of-day using +/-7 days,
     then +/-14 days fallback
4) Resample to 15-minute frequency.
5) Create time, lag and rolling features using past values only.
6) Chronological train/validation/test split: 70/15/15.
7) Train/evaluate:
   - Seasonal Naive
   - SARIMA
   - XGBoost
   - LSTM lookbacks 60m / 6h / 24h
   - Weighted XGBoost + LSTM-24h ensemble
   - Residual hybrid: XGBoost -> residual -> LSTM -> corrected forecast
8) Export predictions, metrics and cleaned/featured datasets.

Important:
- All scalers are fitted only on the train split.
- Rolling features use shift(1) before rolling to avoid target leakage.
- Use --strict-forecast for a deployment-realistic XGBoost setup that excludes
  contemporaneous electrical covariates which may not be known at forecast time.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import statsmodels.api as sm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

NUMERIC_COLS = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]
TARGET = "global_active_power"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Household electricity forecasting benchmark")
    p.add_argument("--input", required=True, help="Raw input CSV")
    p.add_argument("--output-dir", default="outputs", help="Output directory")
    p.add_argument("--start-date", default="2007-01-01", help="Date for first 00:00 row")
    p.add_argument("--freq", default="15min", help="Benchmark frequency; default 15min")
    p.add_argument("--strict-forecast", action="store_true",
                   help="Exclude contemporaneous voltage/reactive/submeter features from XGBoost")
    p.add_argument("--epochs", type=int, default=10, help="Maximum LSTM epochs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace(["?", "NA", "NaN", "nan", ""], np.nan), errors="coerce")


def load_and_restore_datetime(path: Path, start_date: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ["time"] + NUMERIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in NUMERIC_COLS:
        df[c] = _to_numeric(df[c])

    start = pd.Timestamp(start_date)
    # The supplied dataset is minute ordered and contains 1,440 rows/day.
    df["datetime"] = pd.date_range(start=start, periods=len(df), freq="1min")
    df["date"] = df["datetime"].dt.date.astype(str)
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    return df


def detect_missing_gaps(df: pd.DataFrame) -> List[Tuple[int, int, int]]:
    mask = df[NUMERIC_COLS].isna().any(axis=1).to_numpy()
    gaps: List[Tuple[int, int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        s = i
        while i + 1 < n and mask[i + 1]:
            i += 1
        e = i
        gaps.append((s, e, e - s + 1))
        i += 1
    return gaps


def _local_median(df: pd.DataFrame, idx: int, col: str, radius: int = 60) -> float:
    lo, hi = max(0, idx - radius), min(len(df), idx + radius + 1)
    vals = df.loc[lo:hi - 1, col].dropna().to_numpy()
    return float(np.median(vals)) if len(vals) else np.nan


def _seasonal_values(df: pd.DataFrame, idx: int, col: str, days: int) -> List[float]:
    vals: List[float] = []
    for d in range(1, days + 1):
        for sign in (-1, 1):
            j = idx + sign * d * 1440
            if 0 <= j < len(df):
                v = df.at[j, col]
                if pd.notna(v):
                    vals.append(float(v))
    return vals


def fill_missing(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy().reset_index(drop=True)
    gaps = detect_missing_gaps(df)
    report = []

    for s, e, length in gaps:
        if length <= 60:
            method = "linear_interpolation" if length <= 5 else "linear_interpolation_local_median_fallback"
            for c in NUMERIC_COLS:
                left = df.at[s - 1, c] if s > 0 else np.nan
                right = df.at[e + 1, c] if e + 1 < len(df) else np.nan
                if pd.notna(left) and pd.notna(right):
                    for k, idx in enumerate(range(s, e + 1), start=1):
                        if pd.isna(df.at[idx, c]):
                            df.at[idx, c] = float(left + (right - left) * k / (length + 1))
                else:
                    for idx in range(s, e + 1):
                        if pd.isna(df.at[idx, c]):
                            df.at[idx, c] = _local_median(df, idx, c, radius=60)
        else:
            method = "seasonal_median_same_minute_pm7d_pm14d"
            for idx in range(s, e + 1):
                for c in NUMERIC_COLS:
                    if pd.isna(df.at[idx, c]):
                        vals = _seasonal_values(df, idx, c, 7)
                        if len(vals) < 3:
                            vals = _seasonal_values(df, idx, c, 14)
                        if vals:
                            df.at[idx, c] = float(np.median(vals))
                        else:
                            df.at[idx, c] = _local_median(df, idx, c, radius=1440)

        report.append({
            "gap_start": df.at[s, "datetime"],
            "gap_end": df.at[e, "datetime"],
            "gap_minutes": length,
            "method": method,
        })

    if df[NUMERIC_COLS].isna().any().any():
        raise RuntimeError("Missing values remain after imputation")
    return df, pd.DataFrame(report)


def resample_and_features(df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    base_cols = [
        "global_active_power", "global_reactive_power", "voltage",
        "sub_metering_1", "sub_metering_2", "sub_metering_3",
    ]
    x = df.set_index("datetime")[base_cols].resample(freq).mean().dropna().copy()

    x["hour"] = x.index.hour
    x["day_of_week"] = x.index.dayofweek
    x["month"] = x.index.month
    x["is_weekend"] = (x["day_of_week"] >= 5).astype(int)
    x["hour_sin"] = np.sin(2 * np.pi * x["hour"] / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x["hour"] / 24)
    x["dow_sin"] = np.sin(2 * np.pi * x["day_of_week"] / 7)
    x["dow_cos"] = np.cos(2 * np.pi * x["day_of_week"] / 7)

    # For 15-min data: 1=15m, 4=1h, 96=24h, 192=48h, 672=7d
    for lag in [1, 4, 96, 192, 672]:
        x[f"lag_{lag}"] = x[TARGET].shift(lag)

    past = x[TARGET].shift(1)
    for w in [4, 12, 96]:
        x[f"roll_mean_{w}"] = past.rolling(w).mean()
        x[f"roll_std_{w}"] = past.rolling(w).std()

    n = len(x)
    i1, i2 = int(n * 0.70), int(n * 0.85)
    x["split"] = "test"
    x.iloc[:i1, x.columns.get_loc("split")] = "train"
    x.iloc[i1:i2, x.columns.get_loc("split")] = "validation"
    return x


def metric_dict(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mae = mean_absolute_error(y, p)
    rmse = mean_squared_error(y, p) ** 0.5
    mask = np.abs(y) > 1e-6
    mape = np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100
    denom = np.abs(y) + np.abs(p)
    sm = denom > 1e-12
    smape = np.mean(2 * np.abs(y[sm] - p[sm]) / denom[sm]) * 100
    wape = np.sum(np.abs(y - p)) / np.sum(np.abs(y)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE_pct": mape, "sMAPE_pct": smape, "WAPE_pct": wape}


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self) -> int:
        return len(self.X)
    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.lstm(x)
        return self.fc(o[:, -1]).squeeze(-1)


def train_lstm_sequence(
    df: pd.DataFrame,
    lookback: int,
    feature_cols: List[str],
    target_col: str,
    i1: int,
    i2: int,
    max_epochs: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    arr = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df[[target_col]].to_numpy(dtype=np.float32)

    x_scaler = StandardScaler().fit(arr[:i1])
    y_scaler = StandardScaler().fit(y_raw[:i1])
    arr_s = x_scaler.transform(arr).astype(np.float32)
    y_s = y_scaler.transform(y_raw).ravel().astype(np.float32)

    def make_seq(start: int, end: int):
        X, y, idx = [], [], []
        for j in range(max(start, lookback), end):
            X.append(arr_s[j - lookback:j])
            y.append(y_s[j])
            idx.append(j)
        return np.asarray(X, np.float32), np.asarray(y, np.float32), np.asarray(idx)

    Xtr, ytr, _ = make_seq(0, i1)
    Xva, yva, _ = make_seq(i1, i2)
    Xte, _, ite = make_seq(i2, len(df))

    torch.manual_seed(seed)
    model = LSTMRegressor(Xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    tr_loader = DataLoader(SeqDataset(Xtr, ytr), batch_size=256, shuffle=True)
    va_loader = DataLoader(SeqDataset(Xva, yva), batch_size=512, shuffle=False)

    best_state = None
    best_val = np.inf
    bad = 0
    epochs_done = 0

    for ep in range(max_epochs):
        model.train()
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in va_loader:
                vals.append(loss_fn(model(xb), yb).item())
        v = float(np.mean(vals))
        epochs_done = ep + 1
        if v < best_val - 1e-5:
            best_val = v
            bad = 0
            best_state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.from_numpy(Xte)).numpy().reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_s).ravel()
    actual = df[target_col].to_numpy()[ite]
    return actual, pred, ite, best_val, epochs_done


def run_benchmark(featured: pd.DataFrame, strict_forecast: bool, max_epochs: int, seed: int):
    n = len(featured)
    i1, i2 = int(n * 0.70), int(n * 0.85)
    results = []
    predictions = pd.DataFrame(index=featured.index[i2:])
    y_test = featured[TARGET].iloc[i2:].to_numpy()
    predictions["actual"] = y_test

    # Seasonal Naive
    pred_sn = featured[TARGET].shift(96).iloc[i2:].to_numpy()
    results.append({"model": "Seasonal Naive (t-24h)", **metric_dict(y_test, pred_sn)})
    predictions["seasonal_naive"] = pred_sn

    # XGBoost
    xgb_features = [
        "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "lag_1", "lag_4", "lag_96", "lag_192", "lag_672",
        "roll_mean_4", "roll_std_4", "roll_mean_12", "roll_std_12", "roll_mean_96", "roll_std_96",
    ]
    if not strict_forecast:
        xgb_features += ["global_reactive_power", "voltage", "sub_metering_1", "sub_metering_2", "sub_metering_3"]

    xd = featured.dropna(subset=xgb_features + [TARGET]).copy()
    train_cut, val_cut = featured.index[i1], featured.index[i2]
    tr = xd[xd.index < train_cut]
    va = xd[(xd.index >= train_cut) & (xd.index < val_cut)]
    te = xd[xd.index >= val_cut]

    xgb = XGBRegressor(
        n_estimators=250, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        objective="reg:squarederror", random_state=seed, n_jobs=4,
    )
    xgb.fit(tr[xgb_features], tr[TARGET], eval_set=[(va[xgb_features], va[TARGET])], verbose=False)
    pred_xgb = xgb.predict(te[xgb_features])
    results.append({"model": "XGBoost", **metric_dict(te[TARGET].to_numpy(), pred_xgb)})
    predictions.loc[te.index, "xgboost"] = pred_xgb

    # SARIMA via seasonal differencing + ARMA(2,1), one-step filtered test prediction
    sdiff = featured[TARGET] - featured[TARGET].shift(96)
    sdiff_train = sdiff.iloc[96:i2].astype(float)
    sar = sm.tsa.statespace.SARIMAX(
        sdiff_train, order=(2, 0, 1), trend="n",
        enforce_stationarity=True, enforce_invertibility=True,
    )
    fit = sar.fit(disp=False, maxiter=40)
    sdiff_full = sdiff.iloc[96:].astype(float)
    sar_full = sm.tsa.statespace.SARIMAX(
        sdiff_full, order=(2, 0, 1), trend="n",
        enforce_stationarity=True, enforce_invertibility=True,
    )
    filtered = sar_full.filter(fit.params)
    pred_diff = filtered.get_prediction(start=i2 - 96, end=len(sdiff_full) - 1, dynamic=False).predicted_mean.to_numpy()
    pred_sar = featured[TARGET].shift(96).iloc[i2:].to_numpy() + pred_diff
    results.append({"model": "SARIMA(2,0,1)(0,1,0,96)", **metric_dict(y_test, pred_sar)})
    predictions["sarima"] = pred_sar

    # LSTMs
    lstm_features = [
        TARGET, "global_reactive_power", "voltage",
        "sub_metering_1", "sub_metering_2", "sub_metering_3",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]
    lstm_outputs = {}
    for label, lb in [("60m", 4), ("6h", 24), ("24h", 96)]:
        actual, pred_lstm, idx, best_val, epochs_done = train_lstm_sequence(
            featured, lb, lstm_features, TARGET, i1, i2, max_epochs, seed
        )
        results.append({"model": f"LSTM (lookback {label})", **metric_dict(actual, pred_lstm)})
        predictions.loc[featured.index[idx], f"lstm_{label}"] = pred_lstm
        lstm_outputs[label] = {"actual": actual, "pred": pred_lstm, "idx": idx,
                               "best_val": best_val, "epochs": epochs_done}

    # Weighted ensemble XGB + LSTM 24h, tuned on validation by retraining a LSTM prediction set is omitted here.
    # We keep the reproducible fixed weight found in validation search from the experiment: 0.95 / 0.05.
    common = predictions.dropna(subset=["xgboost", "lstm_24h", "actual"]).copy()
    common["hybrid_weighted"] = 0.95 * common["xgboost"] + 0.05 * common["lstm_24h"]
    results.append({"model": "Hybrid weighted (0.95 XGB + 0.05 LSTM)",
                    **metric_dict(common["actual"].to_numpy(), common["hybrid_weighted"].to_numpy())})

    return pd.DataFrame(results), predictions, xgb, xgb_features, (tr, va, te, pred_xgb)


def run_residual_hybrid(
    featured: pd.DataFrame,
    xgb: XGBRegressor,
    xgb_features: List[str],
    split_data,
    max_epochs: int,
    seed: int,
):
    tr, va, te, pred_test = split_data
    pred_train = xgb.predict(tr[xgb_features])
    pred_val = xgb.predict(va[xgb_features])

    res_df = pd.concat([
        pd.DataFrame({"actual": tr[TARGET], "xgb_pred": pred_train}, index=tr.index),
        pd.DataFrame({"actual": va[TARGET], "xgb_pred": pred_val}, index=va.index),
        pd.DataFrame({"actual": te[TARGET], "xgb_pred": pred_test}, index=te.index),
    ]).sort_index()
    res_df["residual"] = res_df["actual"] - res_df["xgb_pred"]

    for lag in [1, 4, 96]:
        res_df[f"resid_lag_{lag}"] = res_df["residual"].shift(lag)
    res_df["resid_roll_mean_4"] = res_df["residual"].shift(1).rolling(4).mean()
    res_df["resid_roll_std_4"] = res_df["residual"].shift(1).rolling(4).std()
    res_df["resid_roll_mean_24"] = res_df["residual"].shift(1).rolling(24).mean()
    res_df["resid_roll_std_24"] = res_df["residual"].shift(1).rolling(24).std()
    for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        res_df[c] = featured.loc[res_df.index, c]
    res_df = res_df.dropna().copy()

    train_cut = featured.index[int(len(featured) * 0.70)]
    val_cut = featured.index[int(len(featured) * 0.85)]
    train_end = np.where(res_df.index < train_cut)[0][-1] + 1
    val_end = np.where(res_df.index < val_cut)[0][-1] + 1

    feat = [
        "residual", "resid_lag_1", "resid_lag_4", "resid_lag_96",
        "resid_roll_mean_4", "resid_roll_std_4", "resid_roll_mean_24", "resid_roll_std_24",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]

    arr = res_df[feat].to_numpy(dtype=np.float32)
    y_raw = res_df[["residual"]].to_numpy(dtype=np.float32)
    xs = StandardScaler().fit(arr[:train_end])
    ys = StandardScaler().fit(y_raw[:train_end])
    arr_s = xs.transform(arr).astype(np.float32)
    y_s = ys.transform(y_raw).ravel().astype(np.float32)

    lookback = 96
    def mk(start, end):
        X, y, idx = [], [], []
        for j in range(max(start, lookback), end):
            X.append(arr_s[j-lookback:j]); y.append(y_s[j]); idx.append(j)
        return np.asarray(X,np.float32), np.asarray(y,np.float32), np.asarray(idx)

    Xtr,ytr,_ = mk(0, train_end)
    Xva,yva,_ = mk(train_end, val_end)
    Xte,_,ite = mk(val_end, len(res_df))

    torch.manual_seed(seed)
    model = LSTMRegressor(Xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    tl = DataLoader(SeqDataset(Xtr,ytr), batch_size=256, shuffle=True)
    vl = DataLoader(SeqDataset(Xva,yva), batch_size=512, shuffle=False)

    best_state=None; best_val=np.inf; bad=0; epochs_done=0
    for ep in range(max_epochs):
        model.train()
        for xb,yb in tl:
            opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
        model.eval(); vals=[]
        with torch.no_grad():
            for xb,yb in vl: vals.append(loss_fn(model(xb),yb).item())
        v=float(np.mean(vals)); epochs_done=ep+1
        if v < best_val - 1e-5:
            best_val=v; bad=0
            best_state={k:vv.detach().clone() for k,vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3: break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred_s = model(torch.from_numpy(Xte)).numpy().reshape(-1,1)
    pred_resid = ys.inverse_transform(pred_s).ravel()

    test_rows = res_df.iloc[ite].copy()
    hybrid = test_rows["xgb_pred"].to_numpy() + pred_resid
    out = pd.DataFrame({
        "actual": test_rows["actual"].to_numpy(),
        "xgb_pred": test_rows["xgb_pred"].to_numpy(),
        "true_residual": test_rows["residual"].to_numpy(),
        "lstm_pred_residual": pred_resid,
        "hybrid_pred": hybrid,
    }, index=test_rows.index)
    metrics = {"model": "XGBoost + Residual LSTM", **metric_dict(out["actual"], out["hybrid_pred"])}
    return metrics, out, {"epochs": epochs_done, "best_val_scaled_mse": best_val}


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_and_restore_datetime(Path(args.input), args.start_date)
    clean, gap_report = fill_missing(raw)
    clean.to_csv(outdir / "household_power_cleaned.csv", index=False)
    gap_report.to_csv(outdir / "gap_fill_report.csv", index=False)

    featured = resample_and_features(clean, args.freq)
    featured.to_csv(outdir / "power_15min_featured.csv")

    base_metrics, predictions, xgb, xgb_features, split_data = run_benchmark(
        featured, args.strict_forecast, args.epochs, args.seed
    )
    residual_metrics, residual_pred, residual_meta = run_residual_hybrid(
        featured, xgb, xgb_features, split_data, max_epochs=max(args.epochs, 12), seed=args.seed
    )

    all_metrics = pd.concat([base_metrics, pd.DataFrame([residual_metrics])], ignore_index=True)
    all_metrics = all_metrics.sort_values("MAE").reset_index(drop=True)
    all_metrics.to_csv(outdir / "model_metrics.csv", index=False)
    predictions.to_csv(outdir / "model_predictions.csv")
    residual_pred.to_csv(outdir / "residual_hybrid_predictions.csv")

    meta = {
        "input": str(args.input),
        "start_date": args.start_date,
        "frequency": args.freq,
        "strict_forecast": args.strict_forecast,
        "split": "70/15/15 chronological",
        "residual_lstm": residual_meta,
        "warning": (
            "When strict_forecast=False, XGBoost uses contemporaneous electrical covariates. "
            "This reproduces the exploratory benchmark but may be optimistic for true ahead-of-time deployment."
        ),
    }
    with (outdir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(all_metrics.to_string(index=False))
    print(f"\nOutputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
