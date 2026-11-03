#!/usr/bin/env python3
"""Prepare strictly chronological, time-tie-safe splits for cl-gbdt (SBCL x86-64)."""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
DATA.mkdir(parents=True, exist_ok=True)
path=Path(sys.argv[1]) if len(sys.argv)>1 else DATA/'creditcard.csv'
df=pd.read_csv(path)
features=[f'V{i}' for i in range(1,29)]+['Amount']
expected=['Time']+[f'V{i}' for i in range(1,29)]+['Amount','Class']
if list(df.columns)!=expected: raise ValueError(f'unexpected columns: {list(df.columns)}')
if len(df)!=284807 or df.Class.sum()!=492: raise ValueError('expected 284807 rows and 492 fraud labels')
if df.isna().any().any(): raise ValueError('missing values found')
df=df.sort_values('Time',kind='stable').reset_index(drop=True)
# Boundaries move right to keep transactions with the same timestamp together.
times=df.Time.to_numpy()
i=np.searchsorted(times,times[int(len(df)*.6)],side='right')
j=np.searchsorted(times,times[int(len(df)*.8)],side='right')
assert times[i-1]<times[i] and times[j-1]<times[j]
sets={'train':df.iloc[:i], 'valid':df.iloc[i:j], 'test':df.iloc[j:]}
meta={'rows':len(df),'frauds':int(df.Class.sum()),'columns':len(features),'features':features,
      'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'duplicated_full_rows':int(df.duplicated().sum()),'partitions':{}}
for name,part in sets.items():
    x=part[features].to_numpy(dtype='<f4',copy=True)
    y=part.Class.to_numpy(dtype='<f4',copy=True)
    x.tofile(DATA/f'{name}-x.f32'); y.tofile(DATA/f'{name}-y.f32')
    meta['partitions'][name]={'rows':len(part),'frauds':int(part.Class.sum()),
          'min_time':float(part.Time.min()),'max_time':float(part.Time.max())}
# Detect feature vectors that occurred in earlier splits. Hash-based diagnostic;
# identical predictors may represent repeated transactions, not duplicate records.
feature_hashes={name:pd.util.hash_pandas_object(part[features],index=False).values for name,part in sets.items()}
for name,earlier in [('valid',['train']),('test',['train','valid'])]:
    prior=set(np.concatenate([feature_hashes[x] for x in earlier]))
    overlap=np.fromiter((h in prior for h in feature_hashes[name]),dtype=np.uint8)
    overlap.tofile(DATA/f'{name}-seen.u8')
    meta['partitions'][name]['previous_feature_match_rows']=int(overlap.sum())
    meta['partitions'][name]['previous_feature_match_frauds']=int((overlap*sets[name].Class.to_numpy()).sum())
(DATA/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
(DATA/'sizes.lisp').write_text('('+' '.join(f':{name} {len(part)}' for name,part in sets.items())+f' :features {len(features)})\n')
print(json.dumps(meta,indent=2))
