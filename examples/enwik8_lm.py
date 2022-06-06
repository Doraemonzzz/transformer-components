# coding=utf-8

import zipfile
import numpy as np
import torch

with zipfile.ZipFile('../.data/enwik8/enwik8.zip') as zipfile:
    with zipfile.open('enwik8') as file:
        X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        trX, vaX = np.split(X, [int(90e6)])
        data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)