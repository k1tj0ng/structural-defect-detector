import numpy as np
import pandas as pd
import pickle
import json
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import recall_score, f1_score

STRUCTURES      = ['Decks', 'Pavements', 'Walls']
THRESHOLD_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MIN_F1          = 0.60
SEED            = 42
KNN_MAX_TRAIN   = 6000

np.random.seed(SEED)

# ── Load ──────────────────────────────────────────────────────────────────────
df_train = pd.read_csv('splits/train_split.csv')
df_val   = pd.read_csv('splits/val_split.csv')
X_train  = np.load('features/features_train.npy')
X_val    = np.load('features/features_val.npy')
X_test   = np.load('features/features_test.npy')
print(f'Loaded: X_train={X_train.shape}  X_val={X_val.shape}')

Path('models/classical').mkdir(parents=True, exist_ok=True)

# ── Model factories ───────────────────────────────────────────────────────────
def make_svm(class_weight=None):
    base = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    LinearSVC(class_weight=class_weight, max_iter=5000, C=1.0, random_state=SEED)),
    ])
    return CalibratedClassifierCV(base, cv=3)

def make_knn():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('knn',    KNeighborsClassifier(n_neighbors=11, algorithm='brute',
                                        metric='cosine', n_jobs=-1)),
    ])

def make_lr(class_weight=None):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lr',     LogisticRegression(class_weight=class_weight, max_iter=1000,
                                      C=1.0, random_state=SEED, n_jobs=-1)),
    ])

def subsample(X, y, n):
    rng = np.random.default_rng(SEED)
    idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        take = max(1, int(n * len(cls_idx) / len(y)))
        idx.extend(rng.choice(cls_idx, min(take, len(cls_idx)), replace=False))
    return X[idx], y[idx]

MODELS = {
    'svm': {'make': make_svm, 'knn': False},
    'knn': {'make': make_knn, 'knn': True},
    'lr':  {'make': make_lr,  'knn': False},
}

# ── Stage 1 ───────────────────────────────────────────────────────────────────
y_struct_train = df_train['structure_idx'].values
y_struct_val   = df_val['structure_idx'].values

for key, info in MODELS.items():
    print(f'\nStage 1 [{key}]...', flush=True)
    if info['knn']:
        Xf, yf = subsample(X_train, y_struct_train, KNN_MAX_TRAIN)
        m = info['make']()
    else:
        Xf, yf = X_train, y_struct_train
        m = info['make'](class_weight=None)
    m.fit(Xf, yf)
    acc = (m.predict(X_val) == y_struct_val).mean()
    print(f'  val_acc={acc:.3f}  (n_train={len(yf):,})')
    with open(f'models/classical/{key}_structure_model.pkl', 'wb') as f:
        pickle.dump(m, f)
    print(f'  saved models/classical/{key}_structure_model.pkl')

# ── Stage 2 ───────────────────────────────────────────────────────────────────
all_thresholds = {k: {} for k in MODELS}

for key, info in MODELS.items():
    print(f'\n{"="*50}')
    print(f'Stage 2 [{key}]')
    print(f'{"="*50}')
    for struct in STRUCTURES:
        mask_tr = (df_train['structure'] == struct).values
        mask_vl = (df_val['structure']   == struct).values
        X_tr = X_train[mask_tr]; y_tr = df_train.loc[mask_tr, 'defect_idx'].values
        X_vl = X_val[mask_vl];   y_vl = df_val.loc[mask_vl, 'defect_idx'].values

        if info['knn']:
            Xf, yf = subsample(X_tr, y_tr, KNN_MAX_TRAIN)
            m = info['make']()
        else:
            Xf, yf = X_tr, y_tr
            m = info['make'](class_weight='balanced')

        print(f'  {struct}: {int((yf==1).sum())} pos / {int((yf==0).sum())} neg  n={len(yf):,}',
              flush=True, end=' ')
        m.fit(Xf, yf)
        print('fitted', end=' ')

        with open(f'models/classical/{key}_defect_{struct.lower()}.pkl', 'wb') as f:
            pickle.dump(m, f)
        print('saved')

        vp = m.predict_proba(X_vl)[:, 1]
        best, found = 0.50, False
        for t in THRESHOLD_SWEEP:
            preds = (vp >= t).astype(int)
            rec = recall_score(y_vl, preds, zero_division=0)
            f1  = f1_score(y_vl, preds, zero_division=0)
            print(f'    t={t:.2f}  recall={rec:.3f}  f1={f1:.3f}')
            if f1 >= MIN_F1 and not found:
                best, found = t, True
        if not found:
            f1s = [f1_score(y_vl, (vp >= t).astype(int), zero_division=0)
                   for t in THRESHOLD_SWEEP]
            best = THRESHOLD_SWEEP[int(np.argmax(f1s))]
            print(f'  WARNING: no threshold ≥ F1={MIN_F1}; using best-F1.')
        all_thresholds[key][struct] = best
        print(f'  → threshold={best}')

with open('models/classical/classical_thresholds.json', 'w') as f:
    json.dump(all_thresholds, f, indent=2)

print('\n=== Done ===')
for p in sorted(Path('models/classical').glob('*.pkl')) + \
         [Path('models/classical/classical_thresholds.json')]:
    print(f'  [OK] {p}')
print('\nThresholds:', all_thresholds)
