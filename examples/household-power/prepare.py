"""Turn minute-resolution power data into leakage-safe, 60-minute-ahead features."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
raw = pd.read_csv(DATA / "household_power_consumption.txt", sep=";", na_values="?",
                  usecols=["Date", "Time", "Global_active_power"], low_memory=False)
timestamp = pd.to_datetime(raw.Date + " " + raw.Time, format="%d/%m/%Y %H:%M:%S")
if not timestamp.is_monotonic_increasing or timestamp.duplicated().any():
    raise ValueError("Expected unique, sorted minute timestamps")
power = pd.Series(raw.Global_active_power.to_numpy(dtype=np.float32), index=timestamp).asfreq("min")
cut1 = power.index[int(len(power) * 0.7)]
cut2 = power.index[int(len(power) * 0.85)]
features = pd.DataFrame(index=power.index)
features["now"] = power
features["lag_1min"] = power.shift(1)
features["lag_60min"] = power.shift(60)
features["lag_1440min"] = power.shift(1440)
features["mean_60min"] = power.rolling(60).mean()
features["mean_1440min"] = power.rolling(1440).mean()
features["hour"] = power.index.hour + power.index.minute / 60
features["weekday"] = power.index.dayofweek
features["target"] = power.shift(-60)
features = features.dropna()
feature_cols = [c for c in features.columns if c != "target"]
gap = pd.Timedelta(minutes=60)
parts = {
    "train": features[features.index < cut1 - gap],
    "valid": features[(features.index >= cut1) & (features.index < cut2 - gap)],
    "test": features[features.index >= cut2],
}
for name, part in parts.items():
    if part.empty:
        raise ValueError(f"Empty {name} split")
    np.asarray(part[feature_cols], dtype="<f4", order="C").tofile(DATA / f"{name}-x.f32")
    np.asarray(part.target, dtype="<f4").tofile(DATA / f"{name}-y.f32")
    print(f"{name}: rows={len(part)} origins={part.index.min()}..{part.index.max()}")
mini = parts["train"].iloc[:100_000]
np.asarray(mini[feature_cols], dtype="<f4", order="C").tofile(DATA / "mini-x.f32")
np.asarray(mini.target, dtype="<f4").tofile(DATA / "mini-y.f32")
parts["test"][["now", "target"]].to_csv(DATA / "test-meta.csv", index=False)
(DATA / "sizes.lisp").write_text(
    f"(:train {len(parts['train'])} :valid {len(parts['valid'])} :test {len(parts['test'])} "
    f":features {len(feature_cols)} :mini {len(mini)})\n")
print(f"raw minute rows={len(raw)}, cutoffs={cut1},{cut2}")
