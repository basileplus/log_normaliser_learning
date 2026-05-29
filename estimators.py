import torch

    
def estimate_mean(samples):
    return samples.mean(0)

def estimate_cov(samples):
    mean = estimate_mean(samples)
    return 1/samples.shape[0] * ((samples - mean).T @ (samples - mean))
