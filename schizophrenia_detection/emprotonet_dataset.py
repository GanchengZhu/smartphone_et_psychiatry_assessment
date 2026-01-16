# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class EyeTrackingDataset(Dataset):
    def __init__(self, csv_path, train=True, scaler=None):
        self.data = pd.read_csv(csv_path)
        self.features = self.data.drop('label', axis=1).values
        self.labels = self.data['label'].values
        self.train = train
        if train:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.features = scaler.transform(self.features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]]).squeeze()
        return features, label