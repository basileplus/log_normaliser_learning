from StatisticalModel import NormalDistribution1D
from estimators import estimate_mean, estimate_cov
from ICNN import ICNN
import torch


# Input
T = 1024
eta_set  = torch.rand((T,1))
batch_size = 512
model = ICNN(1,1)
params = list(model.parameters())
lr = torch.ones(T)*1e-3

# Training loop

for t in range(T):
    eta = eta_set[t].detach().requires_grad_(True)  
    A_star = model(eta)
    theta_pred = torch.autograd.grad(A_star, eta, create_graph=True)[0] # theta = grad_eta A*(eta)
    stat_model = NormalDistribution1D(theta=theta_pred)

    batch1 = stat_model.get_samples(batch_size)
    batch2 = stat_model.get_samples(batch_size)
    t1 = stat_model.t(batch1)
    t2 = stat_model.t(batch2)
    mean = estimate_mean(t1)
    cov_mat = estimate_cov(t2)

    v = cov_mat @ (mean - eta)
    grad = torch.autograd.grad(
        outputs=theta_pred,
        inputs=params,
        grad_outputs=2*v,
        allow_unused=True  
    )

    with torch.no_grad():
            for p, g in zip(params, grad):
                if g is not None:
                    p.sub_(lr[t] * g)
    if t%100==0:
       print(f"Step {t}/{T} completed | theta_pred: [{theta_pred[0].item():.4f}] ")
    

# The estimated theta_pred corresponds to mean of our proba distrib and should be = 0.5
