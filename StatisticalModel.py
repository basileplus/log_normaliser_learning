import torch

class StatisticalModel():
    """
    Abstract class for statistical models
    """
    def get_samples(self,n):
        """
        Get n samples from the staticstical model. Returns a tensor
        """
        raise NotImplementedError

    def t(self,x):
        """
        Get sufficient statistic of tensor x
        """
        raise NotImplementedError
    
class NormalDistribution1D(StatisticalModel):
    """
    Define normal distribution with unknown mean and KNOWN variance
    theta = [mean/std**2, -1/(2*std**2)]
    """ 
    def __init__(self,theta,std=torch.tensor(1)):
        self.std = std
        self.mean = theta[0] * (self.std**2)
        self.theta = theta

    def get_samples(self,n):
        return torch.empty(n).normal_(mean=self.mean.item(), std=self.std.item())
    
    def t(self, x):
        return torch.stack([x], dim=-1) 
    
class NormalDistribution1D_unknownStd(StatisticalModel):
    """
    Define normal distribution with unknown mean and UNKNOWN variance
    theta = [mean/std**2, -1/(2*std**2)]
    """ 
    def __init__(self,theta):
        self.std = torch.sqrt(1/(2*theta[1]))
        self.mean = theta[0] * (self.std**2)
        self.theta = theta

    def get_samples(self,n):
        return torch.empty(n).normal_(mean=self.mean.item(), std=self.std.item())
    
    def t(self, x):
        return torch.stack([x,-x**2], dim=-1) 