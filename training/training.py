# This is the training loop for our vanilla gan 
from os.path  import join
from utils.MNISTLoader import MnistDataloader, MNISTDataSet
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from model.Discriminator import Discriminator
from model.Generator import Generator


input_path = 'data\MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')


latent_dim = 128
num_epochs = 10
k=1
batch_size = 64


mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
data = mnist_dataloader.load_data()
x_training = data[0][0]
y_training = data[0][1]

x_test = data[1][0]
y_test = data[1][1]


transform = transforms.Compose([
    transforms.ToTensor()
])

mnist_dataset = MNISTDataSet(x_training, np.ones(len(x_training)), transform=transform)

loader = DataLoader(
    mnist_dataset,           
    batch_size=batch_size,     
    shuffle=True,
)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
discriminator = Discriminator().to(device)
generator = Generator(latent_dim=latent_dim).to(device)

optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

for epoch in range(num_epochs):
  for batch_idx, batch in enumerate(loader):
    optimizer_d.zero_grad() 

    real_images, real_labels = batch
    
    real_images = real_images.to(device)
    latent_vectors = torch.randn(real_images.size(0), latent_dim).to(device)
    real_images = real_images.view(real_images.size(0), -1)
    fake_images = generator(latent_vectors)
    real_scores = discriminator(real_images)
    fake_scores = discriminator(fake_images.detach())



    loss_d = -(torch.log(real_scores) + torch.log(0.9 - fake_scores)).mean()

    loss_d.backward()         
    optimizer_d.step()        

    optimizer_g.zero_grad()
    latent_vectors = torch.randn(real_images.size(0), latent_dim).to(device)
    fake_images = generator(latent_vectors)
    fake_scores = discriminator(fake_images)
    loss_g = -(torch.log(fake_scores)).mean()
    loss_g.backward()
    optimizer_g.step()

    if batch_idx % 100 == 0:
      print(f"Epoch [{epoch}/{num_epochs}] Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")
  torch.save(generator.state_dict(), 'generator.pth')
  torch.save(discriminator.state_dict(), 'discriminator.pth')