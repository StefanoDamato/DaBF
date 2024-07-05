import torch
from tqdm import tqdm

from unet import UNet
from utils import set_device

dvc =  set_device()

class Diffusion():
    def __init__(self, T, S, betas, UNet_dim = 16):
        assert type(T) == int
        self.T = T
        self.S = S
        self.betas = torch.tensor(betas).repeat(T) if type(betas) == float else betas
        self.alphas = 1-self.betas.to(dvc)
        self.alphas_hat = torch.cumprod(self.alphas, axis=0).to(dvc)
        self.sigma = 0.02
        self.normal = torch.distributions.Normal(torch.tensor([0.]), torch.tensor([1.]))
        self.UNet_dim = UNet_dim
        self.UNet = UNet(self.UNet_dim).to(dvc)

    
    def sample_eps(self, size):
        return self.normal.sample(size).to(dvc)
        

    def sample_t(self, B, all_equal = True):
        if all_equal:
            t = torch.randint(0, self.T, torch.Size([1]), device=dvc).repeat(B)
        else:
            t = torch.randint(0, self.T, torch.Size([B]), device = dvc)
        return t
    
    def load_UNet(self, path):
        self.UNet.load_state_dict(torch.load(path))


    def generate(self, num, save_steps = []):

        assert type(num) == int, "'num' has to be an integer"
        num, B, S = torch.tensor([num], device=dvc), 1, self.S

        process = []
        self.UNet.eval()

        xT = self.sample_eps(torch.Size([B*S**2])).reshape((B,1,S,S))
        if len(save_steps) > 0:
            process.append(torch.squeeze(xT.detach().to('cpu')))

        xt = xT

        for t in reversed(range(self.T)):

            z = self.sample_eps(torch.Size([B*S**2])).reshape((B,1,S,S)) if t>=1 else 0
            alpha, alpha_hat, sigma = self.alphas[t], self.alphas_hat[t], self.sigma

            eps_pred = self.UNet(xt, num, torch.tensor([t], device=dvc).repeat(B))

            xt = (xt - eps_pred*(1-alpha)/torch.sqrt(1-alpha_hat))/torch.sqrt(alpha) + sigma*z

            if t in save_steps:
                process.append(torch.squeeze(xt.detach().to('cpu')))

        x0 = torch.squeeze(xt.detach().to('cpu'))
        if len(save_steps) > 0:
            process.append(x0)

        return x0, process
    


    def train(self, trainloader, epochs, optimizer, scheduler, verbose=True):
        
        S = self.S
        alphas_hat_array = self.alphas_hat.repeat_interleave(S**2).view(-1,1,S,S)

        L2_loss = torch.nn.MSELoss(reduction='sum')

        for epoch in range(epochs):
            epoch_loss = 0

            if verbose:
                trainloader = tqdm(trainloader)

            for x0, l in trainloader:

                optimizer.zero_grad()

                B = x0.shape[0]
                x0, l = x0.to(dvc), l.to(dvc)

                t = self.sample_t(B, all_equal=False)
                eps = self.sample_eps(torch.Size([x0.numel()])).reshape_as(x0)
                
                alphas_hat = alphas_hat_array[t]
                x = x0*torch.sqrt(alphas_hat) + eps*torch.sqrt(1-alphas_hat)

                eps_pred  = self.UNet(x, l, t)
                loss = L2_loss(eps, eps_pred)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            
            scheduler.step()

            if verbose:
                print(f'epoch: {epoch+1} \t loss: {epoch_loss}')