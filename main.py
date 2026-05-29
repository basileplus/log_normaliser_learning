from StatisticalModel import NormalDistribution1D
from estimators import estimate_mean, estimate_cov
from ICNN import ICNN
import torch


# Input
T = 1024
nu_set  = torch.rand((T,2))
batch_size = 512
model = ICNN(2,1)
params = list(model.parameters())
lr = torch.ones(T)*1e-3

# Training loop

for t in range(T):
    nu = nu_set[t].detach().requires_grad_(True)   # shape (2,)
    A_star = model(nu)
    theta_pred = torch.autograd.grad(A_star, nu, create_graph=True)[0] # theta = grad_eta A*(eta)
    stat_model = NormalDistribution1D(theta=theta_pred)

    batch1 = stat_model.get_samples(batch_size)
    batch2 = stat_model.get_samples(batch_size)
    t1 = stat_model.t(batch1)
    t2 = stat_model.t(batch2)
    mean = estimate_mean(t1)
    cov_mat = estimate_cov(t2)

    v = cov_mat @ (mean - nu)
    grad = torch.autograd.grad(
        outputs=theta_pred,
        inputs=params,
        grad_outputs=2*v,
        allow_unused=True  
    )

    with torch.no_grad():
            for p, g in zip(params, grad):
                if g is not None: # We surely have some none gradients because of clamping in StatisticalModel
                    p.sub_(lr[t] * g)
    # if t%100==0:
    #    print(f"Step {t}/{T} completed | theta_pred: [{theta_pred[0].item():.4f}, {theta_pred[1].item():.4f}] ")
    


