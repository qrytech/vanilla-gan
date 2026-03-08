import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):

    def __init__(self, latent_dim = 128):
        super(Generator, self).__init__()


        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 784)


    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        f3 = F.relu(self.fc3(f2))
        output = F.sigmoid(self.fc4(f3))

        return output

