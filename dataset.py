from torch.utils.data import Dataset
from torchvision import transforms
from torch import nn
from PIL import Image
from PIL import ImageFile
import pandas as pd
import torch
from random import random
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassifierDataset(Dataset):

    def __init__(self, dataframe, root_dir, num_classes, transform=None, augmentation=None, p_augmentation=0.5):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            augmentation (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.augmentation = augmentation
        self.num_classes = num_classes
        self.data = dataframe
        self.p_augmentation = 0.5

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
        if self.augmentation and random() < self.p_augmentation:
            try:
                sample['image'] = self.augmentation(sample['image'])
            except: pass
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample


def train_val_split(df, split_ration):
    count_target = {}
    train_indexes = []
    val_indexes = []
    for target in df.target.unique():
        samples = df[df.target == target]
        ln = len(samples)
        samples_indexes = list(samples.index)
        if ln < 25:
            continue
        else:
            random.shuffle(samples_indexes)
            tr_ln = int(ln * split_ration)
            train_indexes += samples_indexes[:tr_ln]
            val_indexes += samples_indexes[tr_ln:]       
    return (
        df.iloc[train_indexes].reset_index(drop=True),
        df.iloc[val_indexes].reset_index(drop=True)
           )


# fixes class indexing
# returns fixed df and num_classes
def fix_df(df):
    # Convert class sequence [0, 1, 3, .., n] to [0, 1, 2, .., num_classes-1]
    coding_cls = { cl: i for i, cl in enumerate(df.target.unique())}
    df.target = df.target.map(lambda x: coding_cls[x])
    num_classes = max(df.target) + 1
    
    return df, num_classes


def create_dataloader(csv_file, root_dir, split_ratio=0.8):
    aug = [
        transforms.ColorJitter(
                brightness=(0.4, 1), 
                contrast=(0.27, 1),
                saturation=(0.35, 1),
                hue=(-0.5, 0.5)),
        transforms.RandomAffine((-10, 10), scale=(0.5 ,1)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation((-10, 10)),
    ]
    transformations = [
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.RandomErasing(
                p=0.15, 
                scale=(0.0065, 0.0585), 
                ratio=(0.1, 6)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    aug = transforms.Compose(aug)
    transformations = transforms.Compose(transformations)
    
    df = pd.read_csv(csv_file, names=['filename', 'target'])[:-1]
    df.target = df.target.map(int)
    # Fix dataframe before split to prevent class inconsistency
    df_train, df_val = train_val_split(df, split_ratio)
    df_train, num_classes = fix_df(df_train)
    df_val, _ = fix_df(df_val)
    
    train_set = ClassifierDataset(
        dataframe=df_train,
        root_dir=root_dir,
        num_classes=num_classes,
        transform=transformations,
        augmentation=aug,
    )
    val_set = ClassifierDataset(
        dataframe=df_val,
        root_dir=root_dir,
        num_classes=num_classes,
        transform=transformations,
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=8)

    return train_loader, val_loader, num_classes
