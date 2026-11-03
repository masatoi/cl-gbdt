"""Construct rolling customer snapshots from UCI Online Retail transactions."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
raw = pd.read_excel(DATA / "Online Retail.xlsx")
raw = raw.dropna(subset=["CustomerID", "InvoiceDate", "InvoiceNo", "Country", "StockCode"])
raw = raw[(raw.Quantity > 0) & (raw.UnitPrice > 0) &
          ~raw.InvoiceNo.astype(str).str.startswith("C")].copy()
raw["CustomerID"] = raw.CustomerID.astype("int64")
raw["amount"] = raw.Quantity * raw.UnitPrice
raw = raw.sort_values("InvoiceDate")
dates = pd.to_datetime(["2011-04-01", "2011-05-01", "2011-06-01", "2011-07-01",
                        "2011-08-01", "2011-09-01", "2011-11-01"])
if not (dates[4] + pd.Timedelta(days=30) <= dates[5] and
        dates[5] + pd.Timedelta(days=30) <= dates[6]):
    raise ValueError("Earlier split's labels overlap the next split's observation date")


def snapshot(when):
    history = raw[(raw.InvoiceDate >= when - pd.Timedelta(days=90)) & (raw.InvoiceDate < when)]
    future = raw[(raw.InvoiceDate >= when) & (raw.InvoiceDate < when + pd.Timedelta(days=30))]
    if when + pd.Timedelta(days=30) > raw.InvoiceDate.max():
        raise ValueError("Insufficient observation horizon")
    grouped = history.groupby("CustomerID")
    x = grouped.agg(last=("InvoiceDate", "max"), invoices=("InvoiceNo", "nunique"),
                    spend=("amount", "sum"), quantity=("Quantity", "sum"),
                    products=("StockCode", "nunique"), mean_price=("UnitPrice", "mean"),
                    country=("Country", "last"))
    x["recency"] = (when - x.pop("last")).dt.total_seconds() / 86400
    recent = history[history.InvoiceDate >= when - pd.Timedelta(days=30)]
    x["recent_spend"] = recent.groupby("CustomerID").amount.sum().reindex(x.index, fill_value=0)
    x["target"] = x.index.isin(future.CustomerID).astype(int)
    x["snapshot"] = when
    return x


snapshots = [snapshot(date) for date in dates]
parts = {"train": pd.concat(snapshots[:5]), "valid": snapshots[5], "test": snapshots[6]}
country_map = {country: i + 1 for i, country in enumerate(sorted(parts["train"].country.unique()))}
columns = ["recency", "invoices", "spend", "quantity", "products", "mean_price",
           "recent_spend", "country"]
for name, part in parts.items():
    x = part[columns].copy()
    x["country"] = x.country.map(country_map).fillna(0)
    np.asarray(x, dtype="<f4", order="C").tofile(DATA / f"{name}-x.f32")
    np.asarray(part.target, dtype="<f4").tofile(DATA / f"{name}-y.f32")
    print(f"{name}: rows={len(part)} positives={part.target.sum()}")
frequency = parts["train"].groupby(level=0).target.transform("size")
weights = 1.0 / frequency.to_numpy()
weights /= weights.mean()
per_customer = pd.Series(weights, index=parts["train"].index).groupby(level=0).sum()
if not np.allclose(per_customer.to_numpy(), per_customer.iloc[0]):
    raise ValueError("Customer training weights do not sum equally")
np.asarray(weights, dtype="<f4").tofile(DATA / "train-w.f32")
for name in ("valid", "test"):
    parts[name][["target"]].to_csv(DATA / f"{name}-meta.csv", index=False)
(DATA / "sizes.lisp").write_text(
    f"(:train {len(parts['train'])} :valid {len(parts['valid'])} :test {len(parts['test'])} "
    f":features {len(columns)} :categorical (7))\n")
print(f"positive transactions after filtering={len(raw)}")
