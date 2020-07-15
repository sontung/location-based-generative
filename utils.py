import torch

def compute_grad(model_):
    res = 0
    for param in model_.parameters():
        if param.requires_grad:
            if param.grad is not None:
                res += abs(torch.sum(torch.abs(param.grad)).item())
    return res