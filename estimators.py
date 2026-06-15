import torch

    
def estimate_mean(samples):
    return samples.mean(dim=1)  

def estimate_cov(samples):
    mean = samples.mean(dim=1, keepdim=True)
    centered = samples - mean
    return 1/(samples.shape[1]-1) * (centered.mT @ centered)
