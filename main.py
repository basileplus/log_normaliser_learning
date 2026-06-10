from StatisticalModel import NormalDistribution1D, NormalDistribution1D_unknownStd
from estimators import estimate_mean, estimate_cov
from ICNN import ICNN
from Visualizer import ConjugacyVisualizer
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch


# Input
T = 2048
mu = torch.randn(T) -3.5
var = 0.01 + torch.rand(T) 
eta_set = torch.stack([mu,-(mu**2 + var)],dim=1) # expectation parameters corresponding to ~N(mu,std²)
batch_size = 64
num_epoch=1

data = TensorDataset(eta_set)
loader = DataLoader(
     dataset=data,
     batch_size=batch_size,
     shuffle=True
)

model = ICNN(2,1)
params = list(model.parameters())
lr = 1e-3
visu = False # set a True to save a gif

eta_probe = torch.zeros((len(eta_set), eta_set.shape[1]))
eta_probe[:, 0] = eta_set[:, 0]
conjVis = ConjugacyVisualizer(eta_probe)

# Training loop
for epoch in range(num_epoch):
    for (eta_batch,) in loader:
        
        eta_batch = eta_batch.detach().requires_grad_(True)  
        A_star = model(eta_batch).squeeze()
        theta_pred = torch.autograd.grad(
            outputs=A_star.sum(), 
            inputs=eta_batch, 
            create_graph=True
        )[0] # theta = grad_eta A*(eta)
        theta_valid = torch.stack([theta_pred[:,0],F.softplus(theta_pred[:,1])], dim=1) # ensures 1/std**2 > 0
        stat_model = NormalDistribution1D_unknownStd(theta=theta_valid)

        # if (visu) and (t % 10 == 0):
        #     conjVis.log(model)

        if torch.isnan(theta_valid).any() :
            break

        batch1 = stat_model.get_samples(batch_size)
        batch2 = stat_model.get_samples(batch_size)
        t1 = stat_model.t(batch1)
        t2 = stat_model.t(batch2)
        mean = estimate_mean(t1)
        cov_mat = estimate_cov(t2)

        v = (cov_mat @ (mean - eta_batch).unsqueeze(-1)).squeeze(-1)

        # Clamp v, prevent gradient explosions occuring from cov estimate (var X^2~sigma^4)
        v_norm = v.norm()
        if v_norm > 10:
            v = 10 * v / v_norm

        grad = torch.autograd.grad(
            outputs=theta_valid,
            inputs=params,
            grad_outputs=2*v,
            allow_unused=True  
        )

        grad_norm = 0
        with torch.no_grad():
                for p, g in zip(params, grad):
                    if g is not None:
                        if g is not None and torch.isnan(g).any():
                            print("NaN détecté dans le gradient")
                        grad_norm += g.norm().item()**2
                        p.sub_(lr * g)
        # if t%100==0:
        #     std = torch.sqrt(1/(2*theta_valid[1]))
        #     mean = theta_valid[0] * (std**2)
        #     print(f"Step {t}/{T} completed | grad norm: {grad_norm**0.5}")
    

if visu:
    conjVis.save_gif()