#!/usr/bin/env python3
"""Prepare fixed train/validation/test splits as inert Lisp-readable numeric data."""
from pathlib import Path
import json
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
import sklearn

ROOT = Path(__file__).resolve().parent
(ROOT / 'data').mkdir(parents=True, exist_ok=True)

def lisp_float(x):
    return format(float(x), '.17e').replace('e', 'd')

def write_data(name, data, classification):
    X, y = data.data, data.target
    if name == 'breast_cancer':
        y = 1 - y  # malignant=1, benign=0 (opposite of sklearn's default)
    index = list(range(len(y)))
    train, hold = train_test_split(index, test_size=0.4, random_state=42,
                                   stratify=y if classification else None)
    valid, test = train_test_split(hold, test_size=0.5, random_state=42,
                                   stratify=y[hold] if classification else None)
    path = ROOT / 'data' / f'{name}.sexp'
    with path.open('w', encoding='utf-8') as f:
        f.write('(')
        for key, indices in [('train', train), ('valid', valid), ('test', test)]:
            f.write(f' :{key} (\n')
            for i in indices:
                f.write('(' + ' '.join([lisp_float(y[i])] + [lisp_float(x) for x in X[i]]) + ')\n')
            f.write(')\n')
        f.write(' :feature-names (' + ' '.join(json.dumps(str(n)) for n in data.feature_names) + '))\n')
    print(f'{name}: train={len(train)} valid={len(valid)} test={len(test)} features={X.shape[1]} -> {path}')

print('scikit-learn', sklearn.__version__)
write_data('iris', load_iris(), True)
write_data('breast_cancer', load_breast_cancer(), True)
write_data('diabetes', load_diabetes(scaled=False), False)
