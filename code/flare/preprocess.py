import numpy as np
import joblib, os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def fit_scalers(x=None, y=None, pca_keep_var=0.98, out_dir='data/scalers'):
    os.makedirs(out_dir, exist_ok=True)
    scalers = {}
    if x is not None:
        xs = StandardScaler().fit(x)
        joblib.dump(xs, os.path.join(out_dir, 'x_scaler.joblib'))
        scalers['x'] = xs
    if y is not None:
        ys = StandardScaler().fit(y)
        joblib.dump(ys, os.path.join(out_dir, 'y_scaler.joblib'))
        scalers['y'] = ys
        scalers['y_pca'] = None
    return scalers

def transform_xy(x=None, y=None, scalers_dir='data/scalers'):
    xs = joblib.load(os.path.join(scalers_dir, 'x_scaler.joblib')) if x is not None and os.path.exists(os.path.join(scalers_dir, 'x_scaler.joblib')) else None
    ys = joblib.load(os.path.join(scalers_dir, 'y_scaler.joblib')) if y is not None else None
    x_t = xs.transform(x) if (x is not None and xs is not None) else None
    if y is not None:
        y_std = ys.transform(y)
        y_t = y_std
    else:
        y_t = None
    return x_t, y_t
