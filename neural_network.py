import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.integrate import odeint
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

N = 10000000

def diff_sird(q, t, N, beta, gamma, mu):
    S, I, R, D = q
    dSdt = -beta * I * S / N
    dIdt = beta * I * S / N - (gamma + mu) * I
    dRdt = gamma * I
    dDdt = mu * I
    return dSdt, dIdt, dRdt, dDdt

class SIRD(nn.Module):
    def forward(self, alpha):
        for i, alpha_t in enumerate(alpha):
            result = odeint(diff_sird, q0, np.arange(alpha.shape[0]), args = (N, *alpha_t))
            result = torch.from_numpy(result)[:, [1, 3]].clone().unsqueeze(dim = 0)
            if i == 0:
                results = result
            else:
                results = torch.cat((results, result), dim = 0)
        return results

class CovidModel(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.fc = nn.Linear(2, 8)
        self.output = nn.Linear(8, 3)
        self.sird = SIRD()
    def forward(self, x):
        alpha = torch.sigmoid(self.output(torch.sigmoid(self.fc(x))))
        return alpha, self.sird(alpha)
def criterion(alpha, alpha_0, q_pred, q_true):
    loss_1 = torch.square(torch.log(q_pred) - torch.log(q_true)).sum()
    loss_2 = 1e-2 * torch.log(torch.tensor(q_true.numpy().max(axis = 0)[0])) * torch.square(q_pred - q_true).sum()
    diff = alpha[1:] - alpha[: -1]
    square_diff = (diff ** 2) * torch.Tensor([1, 1, 100])
    loss_3 = 100 * torch.log(torch.tensor(q_true.numpy().max(axis = 0)[0])) / alpha_0.max() * square_diff.sum()
    diff = alpha[0] - alpha_0
    square_diff = (diff ** 2) * torch.Tensor([1, 1, 100])
    loss_4 = 100 * torch.log(torch.tensor(q_true.numpy().max(axis = 0)[0])) / alpha_0.max() * square_diff.sum()
    return loss_1 + loss_2 + loss_3 + loss_4

def train(i, data, alpha_0, model, optimizer):
    alpha, q_pred = model(data)
    alpha_0 = torch.from_numpy(alpha_0)
    def closure():
        optimizer.zero_grad()
        loss = criterion(alpha, alpha_0, q_pred, data)
        if i == 0:
            loss.backward(retain_graph = True)
        else:
            loss.backward()
        return loss.detach().numpy()
    return optimizer.step(closure)

class CovidDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        item = self.data[index]
        return torch.tensor(item).float()

data = pd.read_csv('data.csv').values[:, 1:]
q0 = (N - 1, 1, 0, 0)
INIT_LR = 1e-3
model = CovidModel(np.array([1, 1e-5, 1e-5]))
optimizer = optim.LBFGS(model.parameters(), lr = INIT_LR)
EPOCHS = 10
bar = tqdm(range(EPOCHS))
train_loader = DataLoader(CovidDataset(data), batch_size = data.shape[0])
for epoch in bar:
    for batch_data in train_loader:
        loss = train(epoch, batch_data, np.array([1, 1e-5, 1e-5]), model, optimizer)
        bar.set_description(f'{epoch + 1}/{EPOCHS}: {loss:.4f}')