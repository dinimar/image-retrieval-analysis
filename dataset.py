from torch.utils.data import Dataset
from torch import nn
from PIL import Image
import pandas as pd
import torch 
import os

class ClassifierDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, num_classes=2, transform=None, m_transformations=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_classes = num_classes
        self.data = pd.read_csv(csv_file, names=['filename', 'label'])[:-1]
        self.root_dir = root_dir
        self.transform = transform
        self.m_transformations = m_transformations

    def __len__(self):
        return len(self.data)
    
    def one_hot_encoder(self, num_class):
        return nn.functional.one_hot(torch.tensor(num_class),self.num_classes).numpy()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        cls = self.data.iloc[idx, 1:]
        cls = int(cls)
        label = self.one_hot_encoder(cls)
        sample = {'image': image, 'landmarks': label}
        if self.transform:
            try:
                sample['image'] = self.transform(sample['image'])
            except:
                sample['image'] = self.m_transform(sample['image'])
        return sample
