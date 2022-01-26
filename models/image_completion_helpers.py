import numpy as np
import torch

def the_image_grid(height,width):
    xs = np.linspace(0,1, height)
    ys = np.linspace(0,1, width)
    xx, yy = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([xx, yy], axis = -1)).float()
    return grid

def make_context_mask(batch_size, h, w, device, test = False):
    random_matrix = torch.rand(batch_size, device=device).unsqueeze(1).expand(batch_size,h*w)

    mask = torch.rand(( batch_size, h*w),device=device)
    mask = (mask >= random_matrix).float()
    mask = mask.view(batch_size, h, w)
    if test:
        mask[:5,:int(h/2),:] = 1#torch.ones((10,int(h*w/2)),device=device )
    return mask
