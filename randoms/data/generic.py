import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class GenericDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, csv_path, target_column=None):
        """Initializes instance of class StudentsPerformanceDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        self.target_column = target_column
        self.df = pd.read_csv(csv_path)
        self.df = self.df.fillna(self.df.interpolate(method='polynomial', order=2))
        # Save target and predictors
        
        if self.target_column:
            self.X = self.df.drop(self.target_column, axis=1).values
            self.y = self.df[self.target_column].values
        else:
            self.X = self.df.values
            self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        if self.target_column:
            return self.X[idx, :], self.y[idx, :]
        else:
            return self.X[idx, :]
    
    @property
    def shape(self):
        return self.X.shape
