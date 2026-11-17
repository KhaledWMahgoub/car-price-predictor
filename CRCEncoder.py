#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: noob
"""
import zlib
import numpy as np
from category_encoders import TargetEncoder
import pandas as pd

class CRCTargetEncoder(TargetEncoder):
    """
    Target encoding + a tiny CRC32-based offset per category.
    encoded = target_encoded + (crc32(category) / 2**32) * noise_scale
    The offset is deterministic and lies in [0, noise_scale).
    """

    def __init__(self, cols=None, smoothing=100, min_samples_leaf=20, noise_scale=1e-3):
        super().__init__(cols=cols, smoothing=smoothing, min_samples_leaf=min_samples_leaf)
        self.noise_scale = noise_scale

    def _offset(self, s):
        keys = s.astype(str)
        table = {k: zlib.crc32(k.encode()) / 2**32 for k in keys.unique()}
        return keys.map(table).to_numpy() * self.noise_scale

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        self._decoders_ = {}
        for col in self.cols:
            uniq_vals = X[col].drop_duplicates().values
            uniq_df = pd.DataFrame(
                {c: [X[c].iloc[0]] * len(uniq_vals) for c in X.columns}
            )
            uniq_df[col] = uniq_vals
            uniq_df = uniq_df.reset_index(drop=True)
            vals = self.transform(uniq_df)[col].to_numpy()
            order = np.argsort(vals)
            self._decoders_[col] = (vals[order], uniq_vals[order])
        return self

    def transform(self, X, y=None, **kwargs):
        encoded = super().transform(X, y, **kwargs)
        for col in self.cols:
            encoded[col] = encoded[col].to_numpy() + self._offset(X[col])
        return encoded

    def inverse_transform(self, X_enc):
        out = X_enc.copy()
        for col in self.cols:
            vals, cats = self._decoders_[col]
            v = out[col].to_numpy(dtype=float)
            if len(vals) == 1:
                out[col] = cats[0]
                continue
            idx = np.searchsorted(vals, v).clip(1, len(vals) - 1)
            idx = np.where(v - vals[idx - 1] < vals[idx] - v, idx - 1, idx)
            out[col] = cats[idx]
        return out


    
