import json
import random
from pathlib import Path
import torch
import os

from torch.utils.data import Dataset

import numpy as np
import trimesh
import trimesh as tm
from scipy.spatial.transform import Rotation

import utils
import prepare_dataset
import mesh_ops
import mesh

np.random.seed(10)


def load_segment(path):
    with open(path) as f:
        segment = json.load(f)
    raw_labels = np.array(segment['raw_labels']) - 1
    sub_labels = np.array(segment['sub_labels']) - 1
    raw_to_sub = np.array(segment['raw_to_sub'])

    return raw_labels, sub_labels, raw_to_sub

class ClassificationDataset(Dataset):
    def __init__(self, dataroot, train=True, shuffle=False, num_workers=4, augment=[], nb_augments=1, in_memory=False, K=4, features=['points'], path=False, split=True, nb_split=0, reduce=False, ret_paths=False, descriptors=True,loop=False,ACVD=False,common_edge=False):
        # super().__init__(shuffle=shuffle, num_workers=num_workers, keep_numpy_array=True, buffer_size=134217728*2)
        super().__init__()
        self.augment = augment
        self.in_memory = in_memory
        self.dataroot = Path(dataroot)
        self.augments = augment
        self.nb_augments = nb_augments
        self.mode = 'train' if train else 'test'
        self.feats = features#['points']#['area', 'face_angles', 'curvs']
        self.K=K
        self.mesh_paths, self.labels = [],[]
        self.split = split
        self.nb_split = nb_split
        self.max_faces = 0
        self.reduce = reduce
        self.ret_paths = ret_paths
        self.descriptors = descriptors
        self.loop = loop
        self.ACVD = ACVD
        self.common_edge = common_edge

        # self.reduce_size = mesh_ops.MeshPool(4, torch.device('cuda'), target_size=500, points=True)
        if self.mode == 'test':
            self.files_map = {}
            self.files = []
            self.file_label = []
        if nb_split==0:
            prepare_dataset.prepare_dataset_with_split(self)
        else:
            prepare_dataset.prepare_dataset_without_split(self)
        self.path = path


    def __getitem__(self, idx):
        label = self.labels[idx]
        if self.mode == 'test':
            file = self.files[idx]
        mesh_features = np.load(self.mesh_paths[idx], allow_pickle=True)

        # print(mesh_features[2][mesh_features[2] != -1])
        # print(idx)
        # print(self.mesh_paths[idx])

        # return mesh_features[0], mesh_features[1].astype(np.float32), mesh_features[2].astype(np.float32), mesh_features[3].astype(int), np.zeros(0),  label
        if not self.descriptors and self.mode == 'test':
            return mesh_features[0], mesh_features[1], mesh_features[2], np.zeros(0), mesh_features[3], file,  label
        elif not self.descriptors:
            return mesh_features[0], mesh_features[1], mesh_features[2], np.zeros(0), mesh_features[3],  label
        if not self.ret_paths and self.reduce:
            return mesh_features[0], mesh_features[1], mesh_features[2], mesh_features[3], np.zeros(0), mesh_features[-1],file,  label#, mesh_features[4]
        elif not self.ret_paths and self.mode == 'test':
            return mesh_features[0], mesh_features[1].astype(np.float32), mesh_features[2].astype(np.float32), mesh_features[3].astype(int), np.zeros(0), mesh_features[4], file,  label#, mesh_features[4]
        elif not self.ret_paths :
            return mesh_features[0], mesh_features[1].astype(np.float32), mesh_features[2].astype(np.float32), mesh_features[3].astype(int), np.zeros(0), mesh_features[4],  label#, mesh_features[4]
        else:
            return mesh_features[0], mesh_features[1].astype(np.float32), mesh_features[2].astype(np.float32), mesh_features[3].astype(int), np.zeros(0), mesh_features[4], self.mesh_paths[idx], label
        # return mesh_features[0], mesh_features[1], mesh_features[2], np.zeros(0), mesh_features[3],  label#, mesh_features[4]
    
    def __len__(self):
        if self.path:
            return 1
        return self.mesh_paths.shape[0]

class SegmentationDataset(Dataset):
    def __init__(self, dataroot, batch_size, train=True, shuffle=False, num_workers=0, augments=None, in_memory=False):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, keep_numpy_array=True, buffer_size=134217728)
        self.batch_size = batch_size
        self.in_memory = in_memory
        self.dataroot = dataroot

        self.augments = []
        if train and augments:
            self.augments = augments

        self.mode = 'train' if train else 'test'
        self.feats = ['area', 'face_angles', 'curvs', 'center', 'normal']

        self.mesh_paths = []
        self.raw_paths = []
        self.seg_paths = []
        self.browse_dataroot()

        self.set_attrs(total_len=len(self.mesh_paths))

    def browse_dataroot(self):
        for dataset in (Path(self.dataroot) / self.mode).iterdir():
            if not dataset.is_dir():
                continue
            for obj_path in dataset.iterdir():
                if obj_path.suffix == '.obj':
                    obj_name = obj_path.stem
                    seg_path = obj_path.parent / (obj_name + '.json')

                    raw_name = obj_name.rsplit('-', 1)[0]
                    raw_path = list(Path(self.dataroot).glob(f'raw/{raw_name}.*'))[0]
                    self.mesh_paths.append(str(obj_path))
                    self.raw_paths.append(str(raw_path))
                    self.seg_paths.append(str(seg_path))
        self.mesh_paths = np.array(self.mesh_paths)
        self.raw_paths = np.array(self.raw_paths)
        self.seg_paths = np.array(self.seg_paths)

    def __getitem__(self, idx):
        faces, feats, Fs = load_mesh(self.mesh_paths[idx], 
                                     normalize=True, 
                                     augments=self.augments,
                                     request=self.feats)
        raw_labels, sub_labels, raw_to_sub = load_segment(self.seg_paths[idx])

        return faces, feats, Fs, raw_labels, sub_labels, raw_to_sub, self.mesh_paths[idx], self.raw_paths[idx]

class MeshDataset(Dataset):
    def __init__(self, dataroot, train=True, shuffle=False, num_workers=4, augment=[], nb_augments=1, in_memory=False, K=4, features=['points'], path=False, split=True, nb_split=0, reduce=False, ret_paths=False, descriptors=True,loop=False,ACVD=False,common_edge=True,name="DFAUST"):
        # super().__init__(shuffle=shuffle, num_workers=num_workers, keep_numpy_array=True, buffer_size=134217728*2)
        super().__init__()
        self.augment = augment
        self.in_memory = in_memory
        self.dataroot = Path(dataroot)
        self.augments = augment
        self.nb_augments = nb_augments
        self.mode = 'train' if train else 'test'
        self.feats = features#['points']#['area', 'face_angles', 'curvs']
        self.K=K
        self.mesh_paths, self.labels = [],[]
        self.split = split
        self.nb_split = nb_split
        self.max_faces = 0
        self.reduce = reduce
        self.ret_paths = ret_paths
        self.descriptors = descriptors
        self.loop = loop
        self.ACVD = ACVD
        self.common_edge = common_edge
        self.name = name

        if self.mode == 'test':
            self.files_map = {}
            self.files = []
            self.file_label = []
        elif self.name == "shrec":
            prepare_dataset.prepare_dataset_SHREC(self)
        elif self.name == "Manifold":
            prepare_dataset.prepare_dataset_Manifold(self)
        elif self.name == "human_mesh":
            prepare_dataset.prepare_human_mesh(self)
        self.path = path


    def __getitem__(self, idx):
        label = self.labels[idx]
        if self.mode == 'test':
            file = self.files[idx]
        else: file = 0

        mesh_features = np.load(self.mesh_paths[idx], allow_pickle=True)

        return mesh_features, self.descriptors, file, label, idx
    
    def __len__(self):
        if self.path:
            return 1
        return self.mesh_paths.shape[0]
