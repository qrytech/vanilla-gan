from model.Discriminator import Discriminator
from model.Generator import Generator
from os.path  import join
from utils.MNISTLoader import MnistDataloader, MNISTDataSet
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from model.Discriminator import Discriminator
from model.Generator import Generator
import torch.nn as nn
import matplotlib.pyplot as plt
latent_dim = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(latent_dim=latent_dim).to(device)
generator.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
generator.eval()

import matplotlib.pyplot as plt

generator.eval()
with torch.no_grad():
    noise = torch.randn(16, latent_dim).to(device)
    fake_images = generator(noise).cpu().view(-1, 28, 28)
    fake_images = (fake_images + 1) / 2

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(fake_images[i], cmap='gray')
    ax.axis('off')
plt.savefig('generated.png')
plt.show()