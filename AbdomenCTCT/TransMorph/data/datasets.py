import os, glob
import torch, sys
from torch.utils.data import Dataset
# from .data_utils import pkload
import matplotlib.pyplot as plt
import random

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import json

def load_sitk(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def convert_L2R_labels(mask, keep_all_label=False):
    
    if keep_all_label:
        label = 14*[1]
        
    else:
        mask[np.isin(mask, [12, 13])] = 0 # Remove adrenal gland
        label = 12*[1]
        
    return mask, label


class AbdomenCTCTDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.train_path = os.path.join(data_path, "imagesTr")
        self.train_labels = os.path.join(data_path, "labelsTr")
        self.train_idx = list(range(1,31))
        self.transforms = transforms
        dset_min = -1000
        dset_max = 2000
        self.normalize = lambda x: (x - dset_min) / (dset_max - dset_min)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        # path = self.paths[index]
        idx_path = self.train_idx[index]
        # tar_list = self.paths.copy()
        tar_list = self.train_idx.copy()
        tar_list.remove(idx_path)
        random.shuffle(tar_list)
        tar_file = tar_list[0]

        # x = nib.load(os.path.join(self.train_path, "AbdomenCTCT_{:04d}_0000.nii.gz".format(idx_path))).get_fdata().astype(np.float32)
        # x_seg = nib.load(os.path.join(self.train_labels, "AbdomenCTCT_{:04d}_0000.nii.gz".format(idx_path))).get_fdata().astype(np.uint8)

        # y = nib.load(os.path.join(self.train_path, "AbdomenCTCT_{:04d}_0000.nii.gz".format(tar_file))).get_fdata().astype(np.float32)
        # y_seg = nib.load(os.path.join(self.train_labels, "AbdomenCTCT_{:04d}_0000.nii.gz".format(tar_file))).get_fdata().astype(np.uint8)

        x = nib.load(os.path.join(self.train_path, "AbdomenCTCT_{:04d}_0000.nii.gz".format(idx_path))).get_fdata()
        x_seg = nib.load(os.path.join(self.train_labels, "AbdomenCTCT_{:04d}_0000.nii.gz".format(idx_path))).get_fdata()

        y = nib.load(os.path.join(self.train_path, "AbdomenCTCT_{:04d}_0000.nii.gz".format(tar_file))).get_fdata()
        y_seg = nib.load(os.path.join(self.train_labels, "AbdomenCTCT_{:04d}_0000.nii.gz".format(tar_file))).get_fdata()
        
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]

        # x, y = x.transpose(0, 2, 1, 3), y.transpose(0, 2, 1, 3)
        # x_seg, y_seg = x_seg.transpose(0, 2, 1, 3), y_seg.transpose(0, 2, 1, 3)

        # normalize data
        x = self.normalize(x)
        y = self.normalize(y)

        # apply transforms
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.train_idx)


class AbdomenCTCTInferDataset(Dataset):
    def __init__(self, data_path, transforms):

        with open(os.path.join(data_path, "AbdomenCTCT_dataset.json"), 'r') as f:
            self.dataset = json.load(f)
        self.val_files = self.dataset["registration_val"]
        
        self.paths = data_path
        self.transforms = transforms
        dset_min = -1000
        dset_max = 2000
        self.normalize = lambda x: (x - dset_min) / (dset_max - dset_min)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        
        idx_files = self.val_files[index]
        # tar_list = self.paths.copy()

        x = nib.load(os.path.join(self.paths, idx_files["fixed"])).get_fdata()
        x_seg = nib.load(os.path.join(self.paths, "labelsTr", idx_files["fixed"].split("/")[-1])).get_fdata()

        y = nib.load(os.path.join(self.paths, idx_files["moving"])).get_fdata()
        y_seg = nib.load(os.path.join(self.paths, "labelsTr", idx_files["moving"].split("/")[-1])).get_fdata()
        
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]

        # normalize data
        x = self.normalize(x)
        y = self.normalize(y)

        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.val_files)

# if __name__ == "__main__":
#     abd = AbdomenCTCTDataset(data_path="/home/animesh/storage/TransMorph_Transformer_for_Medical_Image_Registration/data/AbdomenCTCT", transforms=None)