from torch import nn
import torch

class Projection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(dim,dim,bias=True)

    def forward(self,X):
        return self.projection(X)


class NonLinearProjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,dim)
        )

    def forward(self,X):
        return self.projection(X)


class embDataset(torch.utils.data.Dataset):
  def __init__(self,x,y):
    super().__init__()
    self.x = x
    self.y = y

  def __len__(self):
    return len(self.x)

  def __getitem__(self,idx):
    return self.x[idx], self.y[idx]