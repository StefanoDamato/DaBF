import os
import random
import torch
import torchvision

from diffusion_model import Diffusion
from utils import plot_num, plot_process

DIM = 16
USE_PRETRAINED = True

model_name = 'UNet_dim'+str(DIM)+'_pretrained.pth'
assert model_name in os.listdir(os.path.join(os.path.pardir, 'pretrained_models'))

image_to_tensor = torchvision.transforms.ToTensor()
tensor_to_image = torchvision.transforms.ToPILImage()

mnist_train = torchvision.datasets.MNIST('../data/', train= True, transform=image_to_tensor, download=True)
mnist_test = torchvision.datasets.MNIST('../data/', train= False, transform=image_to_tensor, download=True)

trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(mnist_test, batch_size=128, shuffle=True)


diffusion = Diffusion(T = 100, S = 28, betas = torch.linspace(1e-04, 2e-02, 100), UNet_dim=32)

if USE_PRETRAINED:
    diffusion.load_UNet(model_name)
else:
    optimizer = torch.optim.Adam(diffusion.UNet.parameters(), 1e-03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, 1e-04)
    diffusion.train(trainloader, 50)

num = random.randint(0, 10)
x0, process = diffusion.generate(num, [i*10 for i in range(1, 10)])

plot_num(x0)
plot_process(process)