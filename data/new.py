'''
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd, numpy as np

df = pd.read_csv('data/BEED_Data.csv')
X = df[[f'X{i}' for i in range(1,17)]].values
y = (df['y'] > 0).astype(int).values

rf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print('CV Accuracy:', scores)
print('Mean:', scores.mean())
'''

import pandas as pd
import numpy as np

df = pd.read_csv('d:/Sakshi work/RnD/seizure-detection-xai/data/BEED_Data.csv')
df['y_bin'] = (df['y'] > 0).astype(int)

normal  = df[df['y_bin'] == 0].drop(['y', 'y_bin'], axis=1)
seizure = df[df['y_bin'] == 1].drop(['y', 'y_bin'], axis=1)

print('=== MEAN (average amplitude) ===')
print(f'Normal  : {normal.mean().mean():.4f}')
print(f'Seizure : {seizure.mean().mean():.4f}')

print('\n=== VARIANCE ===')
print(f'Normal  : {normal.var().mean():.4f}')
print(f'Seizure : {seizure.var().mean():.4f}')

print('\n=== STD DEVIATION ===')
print(f'Normal  : {normal.std().mean():.4f}')
print(f'Seizure : {seizure.std().mean():.4f}')

print('\n=== PER CHANNEL VARIANCE ===')
print(f"{'Channel':<6} {'Normal Var':>12} {'Seizure Var':>12} {'Ratio (S/N)':>12}")
for col in normal.columns:
    nv = normal[col].var()
    sv = seizure[col].var()
    print(f'{col:<6} {nv:>12.2f} {sv:>12.2f} {sv/nv:>12.2f}')
