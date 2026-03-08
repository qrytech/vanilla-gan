import torch
import torch.nn as nn
import torch.nn.functional as F




class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()


        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,1)
        self.dropout = nn.Dropout(0.30624)

    def forward(self, input):




        f1 = F.relu(self.fc1(input))
        f1 = self.dropout(f1)

        f2 = F.reul(self.fc2(f1))
        f2 = self.dropout(f2)

        output = F.sigmoid(self.fc3(f2))

        return output

