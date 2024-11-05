import os
from typing import Dict, Literal, Tuple
from time import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchaudio

from models.data_model import BatchDict
import warnings
warnings.filterwarnings("ignore")
from torchvision.transforms import Normalize, GaussianBlur

import sys
import numpy.lib.format
import struct

def npload(file):
    if type(file) == str:
        file=open(file,"rb")
    header = file.read(128)
    if not header:
        return None
    descr = str(header[19:25], 'utf-8').replace("'","").replace(" ","")
    shape = tuple(int(num) for num in str(header[60:120], 'utf-8').replace(',)', ')').replace(', }', '').replace('(', '').replace(')', '').split(','))
    datasize = numpy.lib.format.descr_to_dtype(descr).itemsize
    for dimension in shape:
        datasize *= dimension
    return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))

class CoverDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        file_ext: str,
        dataset_path: str,
        data_split: Literal["train", "val", "test"],
        max_size: int,
        max_len: int,
        versions = None,
        val_versions = None,
        use_val_as_test = False,
        use_test = False,
        crop_size = None,
        mixup = False,
        normalize = False,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.file_ext = file_ext
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.max_len = max_len
        self.max_size = max_size
        self.use_test = use_test
        self.versions = versions
        self.val_versions = val_versions
        self.use_val_as_test = use_val_as_test
        self._load_data()
        
        self.rnd_indices = np.random.permutation(len(self.track_ids))
        self.test_rnd_indices = np.random.permutation(len(self.test_track_ids)) if self.use_test else 0
        self.current_index = 0
        self.test_current_index = 0
        self.p_aug = 0.3
        # self.p_test = len(self.test_track_ids) / (len(self.track_ids) + len(self.test_track_ids)) if self.use_test else 0
        # self.p_test = min(self.p_test, 0.5)
        self.f_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
        self.t_masking = torchaudio.transforms.TimeMasking(time_mask_param=10)
        self.blur = GaussianBlur((3,), sigma=(0.1, 2.0))
        self.crop_size = crop_size
        self.mixup = mixup
        self.coef = 0.5
        self.mean = -20
        self.std = 14
        self.normalize = normalize
        

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, index: int) -> BatchDict:
        track_id = self.track_ids[index]
        anchor_cqt = self.tracks.get(track_id, torch.from_numpy(np.random.randn(84, 50)).float())
        if self.crop_size is not None:
            start_crop = np.random.randint(0, self.max_len - self.crop_size)
            anchor_cqt = anchor_cqt[:, start_crop:start_crop+self.crop_size]
        
        if self.data_split == "train":
            clique_id = self.version2clique.loc[track_id, 'clique']
            (pos_id, neg_id, neg_test_id), pos_list = self._triplet_sampling(track_id, clique_id)  
            if not isinstance(pos_id, np.ndarray):
                positive_cqt = self.tracks.get(pos_id, torch.from_numpy(np.random.randn(84, 50)).float())
                positive_label = torch.tensor(self.version2clique.loc[pos_id, 'clique'], dtype=torch.float)
            else:
                positive_cqt1 = self.tracks.get(pos_id[0], torch.from_numpy(np.random.randn(84, 50)).float())
                positive_cqt2 = self.tracks.get(pos_id[1], torch.from_numpy(np.random.randn(84, 50)).float())
                # positive_cqt3 = self.tracks.get(pos_id[2], torch.from_numpy(np.random.randn(84, 50)).float())
                # positive_cqt = self.coef * positive_cqt1 + (1 - self.coef) * positive_cqt1
                positive_cqt = positive_cqt1.clone()

                cut_thr = np.random.uniform(self.coef-0.3, self.coef+0.3)
                positive_cqt[:, int(50*cut_thr):] = positive_cqt2[:, int(50*cut_thr):].clone()
                positive_label = torch.tensor(self.version2clique.loc[pos_id[0], 'clique'], dtype=torch.float)

                # cut_thr = np.random.uniform(self.coef-0.3, self.coef+0.3)
                # anchor_cqt[:, int(50*cut_thr):] = positive_cqt3[:, int(50*cut_thr):].clone()
            

            negative_test_cqt = torch.empty(0)
            negative_test_label = torch.empty(0)
            if not isinstance(neg_id, np.ndarray):
                negative_cqt = self.tracks.get(neg_id, torch.from_numpy(np.random.randn(84, 50)).float())
                negative_label = torch.tensor(self.version2clique.loc[neg_id, 'clique'], dtype=torch.float)
            else:
                negative_cqt1 = self.tracks.get(neg_id[0], torch.from_numpy(np.random.randn(84, 50)).float())
                negative_cqt2 = self.tracks.get(neg_id[1], torch.from_numpy(np.random.randn(84, 50)).float())
                negative_cqt = negative_cqt1.clone()

                cut_thr = np.random.uniform(self.coef-0.3, self.coef+0.3)
                negative_cqt[:, int(50*cut_thr):] = negative_cqt2[:, int(50*cut_thr):].clone()
                negative_label = torch.tensor(self.version2clique.loc[neg_id[0], 'clique'], dtype=torch.float)

            if self.use_test:
                positive_cqt = anchor_cqt.detach().clone()
                positive_cqt = self.f_masking(positive_cqt)
                positive_cqt = self.t_masking(positive_cqt)
                
                negative_test_cqt = self.tracks.get(neg_test_id, torch.from_numpy(np.random.randn(84, 50)).float())
                negative_test_label = torch.tensor(-1, dtype=torch.float)

                negative_cqt = anchor_cqt.detach().clone()
                negative_label = torch.tensor(clique_id, dtype=torch.float)

                anchor_cqt = negative_test_cqt.detach().clone()
                anchor_cqt = self.f_masking(anchor_cqt)
                anchor_cqt = self.t_masking(anchor_cqt)
                # anchor_cqt = self.blur(anchor_cqt[None])[0]
                
                positive_cqt = anchor_cqt.detach().clone()
                positive_cqt = self.f_masking(positive_cqt)
                positive_cqt = self.t_masking(positive_cqt)
                # positive_cqt = self.blur(positive_cqt[None])[0]

            if np.random.choice([0, 1], p=[1-self.p_aug, self.p_aug]):
                anchor_cqt = self.f_masking(anchor_cqt)
                positive_cqt = self.f_masking(positive_cqt)
                negative_cqt = self.f_masking(negative_cqt)
                if self.use_test:
                    negative_test_cqt = self.f_masking(negative_test_cqt)

            if np.random.choice([0, 1], p=[1-self.p_aug, self.p_aug]):
                shift = np.random.randint(1, 10)
                for i in range(84):
                    anchor_cqt[i, :] = torch.roll(anchor_cqt[i, :], shift)
                    positive_cqt[i, :] = torch.roll(positive_cqt[i, :], shift)
                    negative_cqt[i, :] = torch.roll(negative_cqt[i, :], shift)
                    
            if np.random.choice([0, 1], p=[1-self.p_aug, self.p_aug]):
                anchor_cqt = self.t_masking(anchor_cqt)
                positive_cqt = self.t_masking(positive_cqt)
                negative_cqt = self.t_masking(negative_cqt)
                if self.use_test:
                    negative_test_cqt = self.t_masking(negative_test_cqt)

            if self.crop_size is not None:
                start_crop = np.random.randint(0, self.max_len - self.crop_size)
                positive_cqt = positive_cqt[:, start_crop:start_crop+self.crop_size]

                start_crop = np.random.randint(0, self.max_len - self.crop_size)
                negative_cqt = negative_cqt[:, start_crop:start_crop+self.crop_size]

            if self.normalize:
                negative_cqt = (negative_cqt - self.mean) / (2 * self.std)
                positive_cqt = (positive_cqt - self.mean) / (2 * self.std)

        else:
            clique_id = -1
            pos_id = torch.empty(0)
            positive_cqt = torch.empty(0)
            positive_label = torch.empty(0)
            neg_id = torch.empty(0)
            negative_cqt = torch.empty(0)
            negative_label = torch.empty(0)
            negative_test_cqt = torch.empty(0)
            negative_test_label = torch.empty(0)

        if self.normalize:
            anchor_cqt = (anchor_cqt - self.mean) / (2 * self.std)
        
        return dict(
            anchor_id=track_id,
            anchor=anchor_cqt,
            anchor_label=torch.tensor(clique_id, dtype=torch.float),
            positive_id=pos_id,
            positive=positive_cqt,
            positive_label=positive_label,
            negative_id=neg_id,
            negative=negative_cqt,
            negative_label=negative_label,
            negative_test_cqt=negative_test_cqt,
            negative_test_label=negative_test_label,
        )

    def _make_file_path(self, track_id, file_ext):
        a = track_id % 10
        b = track_id // 10 % 10
        c = track_id // 100 % 10
        return os.path.join(str(c), str(b), str(a), f'{track_id}.{file_ext}')

    def _triplet_sampling(self, track_id: int, clique_id: int) -> Tuple[int, int]:
        versions = self.versions.loc[clique_id, "versions"]
        pos_list = np.setdiff1d(versions, track_id)
        pos_id = np.random.choice(pos_list, 1)[0]
        if self.mixup:
            pos_id = np.random.choice(pos_list, 2)
        if self.current_index >= len(self.rnd_indices):
            self.current_index = 0
            self.rnd_indices = np.random.permutation(len(self.track_ids))

        neg_test_id = -1
        if self.use_test:
            neg_test_id = self.test_track_ids[self.test_rnd_indices[self.test_current_index]]
            self.test_current_index += 1
            while neg_test_id in versions:
                if self.test_current_index >= len(self.test_rnd_indices):
                    self.test_current_index = 0
                    self.test_rnd_indices = np.random.permutation(len(self.test_track_ids))
                neg_test_id = self.test_track_ids[self.test_rnd_indices[self.test_current_index]]
                self.test_current_index += 1
            if self.test_current_index >= len(self.test_rnd_indices):
                self.test_current_index = 0
                self.test_rnd_indices = np.random.permutation(len(self.test_track_ids))
                
        neg_id = self.track_ids[self.rnd_indices[self.current_index]]
        self.current_index += 1
        while neg_id in versions:
            if self.current_index >= len(self.rnd_indices):
                self.current_index = 0
                self.rnd_indices = np.random.permutation(len(self.track_ids))
            neg_id = self.track_ids[self.rnd_indices[self.current_index]]
            self.current_index += 1

        if self.mixup:
            
            neg_clique_id = self.version2clique.loc[neg_id, 'clique']
            neg_versions = self.versions.loc[neg_clique_id, "versions"]
            neg_list = np.setdiff1d(neg_versions, neg_id)
            neg_id_2 = np.random.choice(neg_list, 1)[0]
            neg_id = np.array([neg_id, neg_id_2])
            
        return (pos_id, neg_id, neg_test_id), pos_list

    def _load_data(self) -> None:
        if self.data_split in ['train', 'val']:
            mapping = {}
            for k, clique in enumerate(sorted(self.versions["clique"])):
                mapping[clique] = k
            self.versions["clique"] = self.versions["clique"].map(lambda x: mapping[x])
            self.versions.set_index("clique", inplace=True)
            self.version2clique = pd.DataFrame(
                [{'version': version, 'clique': clique} for clique, row in self.versions.iterrows() for version in row['versions']]
            ).set_index('version')
            self.track_ids = self.version2clique.index.to_list()
            if self.use_test:
                if self.use_val_as_test and (self.val_versions is not None):
                    mapping = {}
                    for k, clique in enumerate(sorted(self.val_versions["clique"])):
                        mapping[clique] = k
                    self.val_versions["clique"] = self.val_versions["clique"].map(lambda x: mapping[x])
                    self.val_versions.set_index("clique", inplace=True)
                    self.val_versions2clique = pd.DataFrame(
                        [{'version': version, 'clique': clique} for clique, row in self.val_versions.iterrows() for version in row['versions']]
                    ).set_index('version')
                    self.test_track_ids = self.val_versions2clique.index.to_list()
                else:
                    self.test_track_ids = np.load(os.path.join(self.data_path, "splits", "test_ids.npy"))
                    if self.max_size != -1:
                        self.test_track_ids = self.test_track_ids[:self.max_size]
        else:
            self.track_ids = np.load(os.path.join(self.data_path, "splits", "{}_ids.npy".format(self.data_split)))
            if self.max_size != -1:
                self.track_ids = self.track_ids[:self.max_size]
        
        self.tracks = {}
        for index in tqdm(self.track_ids):
            self.tracks[index] = self._load_cqt(index)

        if self.use_test:
            for index in tqdm(self.test_track_ids):
                if index in self.tracks.keys():
                    print("Error, Intersection")
                if self.use_val_as_test and (self.val_versions is not None):
                    self.tracks[index] = self._load_cqt(index)
                else:
                    self.tracks[index] = self._load_cqt(index, test=True)
        
    def _load_cqt(self, track_id: str, test = False) -> torch.Tensor:
        if test:
            path = self.dataset_path.split('/')
            path[-1] = "test"
            filename = os.path.join('/'.join(path), self._make_file_path(track_id, self.file_ext))
        else:
            filename = os.path.join(self.dataset_path, self._make_file_path(track_id, self.file_ext))
        cqt_spectrogram = np.load(filename)
        # cqt_spectrogram = npload(filename)
        # cqt_spectrogram = np.exp(cqt_spectrogram)
        return torch.from_numpy(cqt_spectrogram)


def cover_dataloader(
    data_path: str,
    file_ext: str,
    dataset_path: str,
    data_split: Literal["train", "val", "test"],
    max_len: int,
    batch_size: int,
    versions,
    val_versions=None,
    **config: Dict,
) -> DataLoader:
    return DataLoader(
        CoverDataset(data_path, file_ext, dataset_path, data_split, versions=versions, val_versions=val_versions, max_size=config.get("max_size", -1), max_len=max_len, use_val_as_test=config["use_val_as_test"], use_test=config["use_test"], crop_size=config.get("crop_size", None), mixup=config.get("mixup", False), normalize=config.get("normalize", False)),
        batch_size=batch_size if max_len > 0 else 1,
        num_workers=config["num_workers"],
        shuffle=config["shuffle"],
        drop_last=config["drop_last"],
        pin_memory=True
    )