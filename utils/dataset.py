import torch
from torch.utils.data import Dataset
import pandas as pd

class StudentDataset(Dataset):

    def __init__(self,file):

        df = pd.read_csv(file)

        self.X = df[['study_time','clicks','quiz_score']].values
        self.y = df['engagement'].values

        self.X = torch.tensor(self.X,dtype=torch.float32)
        self.y = torch.tensor(self.y,dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]