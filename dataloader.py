from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, BatchSampler
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
import argparse
from dataset import *
import itertools

def get_dataloader(dataset, phase, batch_size):
    indices_norm = []
    indices_anorm = []
    for index, data in enumerate(dataset):
        if data['anomaly']:
            indices_anorm.append(index)
        else:
            indices_norm.append(index)
    sub_sampler_norm = SubsetRandomSampler(indices_norm)
    sub_sampler_anorm = SubsetRandomSampler(indices_anorm)
    n_norm = len(sub_sampler_norm)
    n_anorm = len(sub_sampler_anorm)
    if n_norm > n_anorm:
        k = n_norm // n_anorm
        new_indices_anorm = indices_anorm * (k+1)   # When the dataset is small, use circular sampling
        sub_sampler_anorm = SubsetRandomSampler(new_indices_anorm[:n_norm])
    elif n_norm < n_anorm:
        k = n_anorm // n_norm
        new_indices_norm = indices_norm * (k+1)   # When the dataset is small, use circular sampling
        sub_sampler_norm = SubsetRandomSampler(new_indices_norm[:n_anorm])
    sampler_my = BatchSampler_my(sub_sampler_norm, sub_sampler_anorm, batch_size, drop_last=True)
    if phase == 'train':
        data_loader = DataLoader(dataset, batch_sampler=sampler_my)
    else:
        # data_loader = DataLoader(dataset, batch_sampler=sampler_my)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader
