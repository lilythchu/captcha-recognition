import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
from utils import *
import wandb

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
os.makedirs('checkpoints', exist_ok=True)

def real_data_target(size):
    data = Variable(torch.ones(size, 1)).to(device)
    return data

def fake_data_target(size):
    data = Variable(torch.zeros(size, 1)).to(device)
    return data

def imgs_to_vec(imgs, img_size=28):
    return imgs.view(imgs.size(0), img_size * img_size)

def vec_to_imgs(vec):
    return vec.view(vec.size(0), 1, 28, 28)

def noise(size, batch_size=128):
    n = Variable(torch.randn(size, batch_size))
    return n

def make_grid(imgs):
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
    count = 0
    for y in range(4):
        for x in range(4):
            img = imgs[count].view(28, 28)
            axs[y][x].imshow(img, cmap="gray")
            axs[y][x].axis('off')
            count += 1
    return fig
            
def mnist_data():
    return CharsDataset('train.csv', 'train/')

data = mnist_data()

data_loader = DataLoader(data, batch_size=128, shuffle=True)

class DiscriminatorNet(nn.Module):
    """ 3 hidden layer discriminative nn. """
    def __init__(self, n_out=1, img_size=28):
        super(DiscriminatorNet, self).__init__()
        
        self.hidden0 = nn.Sequential(
            nn.Linear(img_size * img_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        self.out = nn.Sequential(
            nn.Linear(256, n_out),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

class GeneratorNet(nn.Module):
    """ 3 hidden layer generative nn. """
    def __init__(self, batch_size=128, img_size=28):
        super(GeneratorNet, self).__init__()
        
        self.hidden0 = nn.Sequential(
            nn.Linear(batch_size, 256),
            nn.LeakyReLU(0.2),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
        )
        self.out = nn.Sequential(
            nn.Linear(1024, img_size * img_size),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


def train_discriminator(dmodel, optimizer, loss_func, real_data, fake_data):
    optimizer.zero_grad()
    real_data.to(device)
    fake_data.to(device)
    # Train on real data
    pred_real = dmodel(real_data)
    error_real = loss_func(pred_real, real_data_target(real_data.size(0)))
    error_real.backward()
    # Train on fake data
    pred_fake = dmodel(fake_data)
    error_fake = loss_func(pred_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    # Update weights
    optimizer.step()
    # Return error
    return error_real + error_fake, pred_real, pred_fake

def train_generator(dmodel, optimizer, loss_func, fake_data):
    optimizer.zero_grad()
    fake_data.to(device)
    pred = dmodel(fake_data)
    error = loss_func(pred, real_data_target(pred.size(0)))
    error.backward()
    optimizer.step()
    return error


if __name__ == '__main__':
    discriminator = DiscriminatorNet().to(device)
    generator = GeneratorNet().to(device)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    loss = nn.BCELoss()

    EPOCHS = 1001
    num_samples = 16
    test_noise = noise(num_samples).to(device)
    test_noise

    wandb.init(project='captcha')
    for epoch in range(EPOCHS):
        for batch_idx, (batch, _) in enumerate(data_loader):
            # === Train D ===
            real_data = Variable(imgs_to_vec(batch)).to(device)
            fake_data = generator(noise(real_data.size(0)).to(device)).detach()
            d_error, d_pred_real, d_pred_fake, = train_discriminator(
                discriminator, d_optimizer, loss, real_data, fake_data)
            # === Train G ===
            fake_data = generator(noise(real_data.size(0)).to(device))
            g_error = train_generator(
                discriminator, g_optimizer, loss, fake_data)
            # === Logging ===
            #log(d_error, g_error, epoch, batch_idx, len(data_loader))
            
        if (epoch % 20) == 0:
            # save checkpoint
            torch.save(generator.state_dict(), f'checkpoints/ckpt_{epoch}.pth')
            # Generate images
            test_images = vec_to_imgs(generator(test_noise)).data.cpu()
            grid_img = make_grid(test_images)
            # Logging
            wandb.log({'D Error': d_error, 'G Error': g_error})
            wandb.log({'Generated Images': [wandb.Image(grid_img)]})
            print(f"Epoch {epoch+1}/{EPOCHS}:"
            f" D Error: {d_error:.4f} G Error: {g_error:.4f}")
