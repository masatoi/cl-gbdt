#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
root=Path(__file__).resolve().parent
out=root/'output'
y=np.fromfile(root/'data/test-y.f32',dtype='<f4').astype(int)
result=json.loads((out/'metrics.json').read_text())
fig,ax=plt.subplots(figsize=(8.2,5.3),layout='constrained')
for model,color in [('lightgbm','#ed8b36'),('xgboost','#2265ae')]:
    p=np.loadtxt(out/f'{model}-test.csv')
    precision,recall,_=precision_recall_curve(y,p)
    ap=result[model]['test_tuned']['average_precision']
    ax.step(recall,precision,where='post',label=f'{model}: AP={ap:.3f}',color=color,lw=1.8)
    m=result[model]['test_tuned']; ax.scatter([m['recall']],[m['precision']],color=color,s=48,zorder=3)
ax.axhline(y.mean(), color='#6b7280', ls='--',lw=1,label=f'Random baseline: {y.mean():.4f}')
ax.set(xlim=(0,1),ylim=(0,1.04),xlabel='Recall (fraud detected / all fraud)',ylabel='Precision (fraud detected / alerts)',title='Credit card fraud: chronological test set')
ax.grid(alpha=.2); ax.legend(loc='lower left');
fig.text(.5,-.015,'Dots: threshold maximizing validation F1. Test: 56,961 rows, 75 fraud.',ha='center',fontsize=9)
fig.savefig(out/'precision-recall.png',dpi=180,bbox_inches='tight')
