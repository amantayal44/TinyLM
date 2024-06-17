import torch

def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    return device

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
