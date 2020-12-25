import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from skimage import io


class CovidDataset(Dataset):
    """
    Covid dataset

    :param root (str): Directory with train/test/val split csv
    :param phase (str): Loads train, test or val set.
    """

    def __init__(self,
                 root: str,
                 phase: str = 'train',
                 transform=None
                 ):
        self.phase = phase
        self.transform = transform
        self.root = root
        self.data = []

        if self.phase not in ['train', 'test', 'val']:
            raise ValueError("Input arg 'phase' must be one of (train, test, val) instead got %s" % phase)

        name = os.path.join(self.root, self.phase + "_set.csv")
        data_split = pd.read_csv(name)
        for idx, row in data_split.iterrows():
            image = io.imread(row['X'], as_gray=True)
            label = str(row['y'])
            self.data.append({'image': image, 'label': label})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        img, label = self.data[idx]['image'], self.data[idx]['label']
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        if isinstance(img, tuple):
            img = np.vstack(img)
        img = img[..., np.newaxis, np.newaxis]
        img = img.transpose((3, 2, 1, 0))
        label = [int(label)]
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.float32)



