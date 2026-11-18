"""Six-task loading with training-only preprocessing and aligned sample IDs."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch

TASK_CLASSES = {'ANM': 3, 'ANM_early': 3, 'ANM_MMSE': 5,
                'ADNI': 3, 'ADNI_early': 3, 'ADNI_MMSE': 4}


class TrainPreprocessor:
    def __init__(self, standardize=False, state=None):
        self.standardize = standardize
        self.state = state

    def fit(self, frame):
        a = frame.to_numpy(dtype=np.float64)
        keep = np.isfinite(a).any(axis=0)
        if not keep.any():
            raise ValueError('No finite training features')
        a = a[:, keep]
        a[~np.isfinite(a)] = np.nan
        medians = np.nanmedian(a, axis=0)
        a = np.where(np.isfinite(a), a, medians)
        mean = a.mean(axis=0) if self.standardize else np.zeros(a.shape[1])
        scale = a.std(axis=0) if self.standardize else np.ones(a.shape[1])
        scale[scale < 1e-12] = 1
        self.state = {'input_columns': list(frame.columns), 'keep': keep.tolist(),
                      'medians': medians.tolist(), 'mean': mean.tolist(), 'scale': scale.tolist(),
                      'standardize': self.standardize}
        return self

    def transform(self, frame):
        s = self.state
        if s is None:
            raise ValueError('Fit preprocessing on training data first')
        if list(frame.columns) != s['input_columns']:
            raise ValueError('Feature columns/order differ from training')
        a = frame.to_numpy(dtype=np.float64)[:, np.asarray(s['keep'], dtype=bool)]
        a = np.where(np.isfinite(a), a, np.asarray(s['medians']))
        result = ((a - np.asarray(s['mean'])) / np.asarray(s['scale'])).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('Non-finite transformed features')
        return result


class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', datapath=None, preprocessor=None,
                 standardize=False, **unused):
        if unused:
            raise TypeError('Use preprocessor= to share training statistics')
        base = Path(datapath)
        plasma = pd.read_csv(base / f'X_{split}_plasma.csv', dtype={'ID_Visit': str})
        labels = pd.read_csv(base / f'y_{split}.csv', dtype={'ID_Visit': str})
        images = pd.read_pickle(base / f'X_{split}_img.pkl')
        if 'ID_Visit' not in plasma or 'ID_Visit' not in labels or 'img_array' not in images:
            raise ValueError('Missing explicit sample ID or img_array column')
        ids = labels['ID_Visit'].tolist()
        if len(ids) != len(set(ids)) or plasma['ID_Visit'].tolist() != ids:
            raise ValueError('Duplicate visits or misaligned plasma/label IDs')
        if 'ID_Visit' in images:
            image_ids = images['ID_Visit'].astype(str).tolist()
        elif images.index.name == 'ID_Visit':
            image_ids = images.index.astype(str).tolist()
        else:
            raise ValueError('MRI has no sample IDs; regenerate data to verify image/label alignment')
        if image_ids != ids:
            raise ValueError('MRI order differs from plasma/labels')
        self.ids = ids
        features = plasma.drop(columns='ID_Visit').replace('.', np.nan).apply(pd.to_numeric, errors='raise')
        if preprocessor is None:
            if split != 'train':
                raise ValueError('Validation/test must receive training-fitted preprocessing')
            preprocessor = TrainPreprocessor(standardize=standardize).fit(features)
        self.preprocessor = preprocessor
        self.csv_data = preprocessor.transform(features)
        self.pkl_data = [np.asarray(v, dtype=np.float32) for v in images['img_array']]
        self.label = labels['label'].to_numpy(dtype=np.int64)
        if not len(ids):
            raise ValueError('Empty split')
        if any(v.shape != (3,72,72) or not np.isfinite(v).all() for v in self.pkl_data):
            raise ValueError('Expected finite MRI arrays of shape (3,72,72); prepare with explicit resize if needed')
        self.fingerprint = hashlib.sha256(('\n'.join(ids) + '\n' + ','.join(map(str, self.label))).encode()).hexdigest()

    def __len__(self): return len(self.ids)
    def get_csvshape(self): return self.csv_data.shape[1]
    def get_true_label(self): return self.label.copy()
    def __getitem__(self, index):
        return (torch.tensor(self.pkl_data[index], dtype=torch.float32),
                torch.tensor(self.csv_data[index], dtype=torch.float32),
                torch.tensor(self.label[index], dtype=torch.long))
