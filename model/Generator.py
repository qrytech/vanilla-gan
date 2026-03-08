import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):

    def __init__(self, latent_dim = 256):
        super(Generator, self).__init__()


        self.fc1 = nn.Linear(latent_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.bn3= nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 784)


    def forward(self, input):
        x = F.leaky_relu(self.bn1(self.fc1(input)), 0.2)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.2)
        x = torch.tanh(self.fc4(x))
        return x
