from StatisticalModel import NormalDistribution1D, NormalDistribution1D_unknownStd
from estimators import estimate_mean, estimate_cov
from ICNN import ICNN
from Visualizer import ConjugacyVisualizer
import torch.nn.functional as F
import torch


# Input
T = 4096
mu = torch.randn(T) -3.5
var = 0.1 + torch.rand(T) 
eta_set = torch.stack([mu,-(mu**2 + var)],dim=1) # expectation parameters corresponding to ~N(mu,std²)
batch_size = 512
model = ICNN(2,1)
params = list(model.parameters())
lr = torch.ones(T)*1e-3
visu = False # set a True to save a gif

eta_probe = torch.zeros((len(eta_set), eta_set.shape[1]))
eta_probe[:, 0] = eta_set[:, 0]
conjVis = ConjugacyVisualizer(eta_probe)

# Training loop
for t in range(T):
    eta = eta_set[t].detach().requires_grad_(True)  
    A_star = model(eta)
    theta_pred = torch.autograd.grad(A_star, eta, create_graph=True)[0] # theta = grad_eta A*(eta)
    theta_valid = torch.stack([theta_pred[0],F.softplus(theta_pred[1])]) # ensures 1/std**2 > 0
    stat_model = NormalDistribution1D_unknownStd(theta=theta_valid)

    if (visu) and (t % 100 == 0):
        conjVis.log(model)

    batch1 = stat_model.get_samples(batch_size)
    batch2 = stat_model.get_samples(batch_size)
    t1 = stat_model.t(batch1)
    t2 = stat_model.t(batch2)
    mean = estimate_mean(t1)
    cov_mat = estimate_cov(t2)

    v = cov_mat @ (mean - eta)

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
                    grad_norm += g.norm().item()**2
                    p.sub_(lr[t] * g)
    if t%100==0:
        std = torch.sqrt(1/(2*theta_valid[1]))
        mean = theta_valid[0] * (std**2)
        print(f"Step {t}/{T} completed | mean pred: {mean} | std pred: {std} | grad norm: {grad_norm**0.5}")
if visu:
    conjVis.save_gif()