import matplotlib.pyplot as plt
import torch

def set_device(dvc = None):
    if dvc is None:
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    return dvc


def plot_num(x, cmap='grey'):
    
    assert x.ndim == 2
    fig = plt.imshow(x, cmap=cmap)
    plt.axis('off')

    return fig

def plot_process(x, space=False, cmap='grey'): 
    fig, axes = plt.subplots(1, len(x), figsize=(2*len(x), len(x)))

    for i, ax in enumerate(axes):
        ax.imshow(x[i], cmap=cmap)
        ax.axis('off')
        if space:
            plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
