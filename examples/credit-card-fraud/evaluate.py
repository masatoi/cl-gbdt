#!/usr/bin/env python3
"""Compute validation-selected threshold and untouched-test metrics from Lisp scores."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, precision_recall_curve, log_loss
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; OUT=ROOT/'output'

def metrics(y, p, t):
    yhat=(p>=t).astype(int)
    tn, fp, fn, tp=confusion_matrix(y,yhat,labels=[0,1]).ravel()
    count=lambda x:int(x)
    precision=float(tp/(tp+fp)) if tp+fp else 0.
    recall=float(tp/(tp+fn)) if tp+fn else 0.
    return {'rows':len(y), 'frauds':int(y.sum()), 'prevalence':float(y.mean()),
            'average_precision':float(average_precision_score(y,p)), 'roc_auc':float(roc_auc_score(y,p)),
            'log_loss':float(log_loss(y,p,labels=[0,1])), 'threshold':float(t),
            'tp':count(tp),'fp':count(fp),'fn':count(fn),'tn':count(tn),
            'precision':precision,'recall':recall,
            'f1':float(2*precision*recall/(precision+recall)) if precision+recall else 0.}

def topk(y,p,k):
    k=min(k,len(y)); ix=np.argsort(-p,kind='stable')[:k]
    tp=int(y[ix].sum()); return {'k':k,'frauds_found':tp,'precision':tp/k,'recall':tp/int(y.sum())}

result={}
for model in ['lightgbm','xgboost']:
    yv=np.fromfile(DATA/'valid-y.f32',dtype='<f4').astype(int)
    yt=np.fromfile(DATA/'test-y.f32',dtype='<f4').astype(int)
    pv=np.loadtxt(OUT/f'{model}-valid.csv'); pt=np.loadtxt(OUT/f'{model}-test.csv')
    assert len(yv)==len(pv) and len(yt)==len(pt)
    assert np.all(np.isfinite(pv)) and np.all(np.isfinite(pt))
    assert np.all((pv>=0)&(pv<=1)) and np.all((pt>=0)&(pt<=1))
    prec,rec,thr=precision_recall_curve(yv,pv)
    f1=np.divide(2*prec[:-1]*rec[:-1],prec[:-1]+rec[:-1],out=np.zeros_like(thr),where=(prec[:-1]+rec[:-1])>0)
    chosen=float(thr[np.argmax(f1)])
    seen=np.fromfile(DATA/'test-seen.u8',dtype=np.uint8).astype(bool)
    result[model]={'validation_selected_f1_threshold':chosen,
                   'validation':metrics(yv,pv,chosen),'test_tuned':metrics(yt,pt,chosen),
                   'test_default':metrics(yt,pt,0.5),'test_top100':topk(yt,pt,100),
                   'test_unseen_features':metrics(yt[~seen],pt[~seen],chosen)}
(OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
for model,r in result.items():
    print('\n==',model,'== threshold from validation:',round(r['validation_selected_f1_threshold'],8))
    for key in ['validation','test_tuned','test_default','test_unseen_features']:
        m=r[key]; print(key, f"n={m['rows']} fraud={m['frauds']} AP={m['average_precision']:.5f} ROC_AUC={m['roc_auc']:.5f}",
                        f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}",f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
    print('test_top100',r['test_top100'])
