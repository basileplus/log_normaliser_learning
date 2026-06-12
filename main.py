from StatisticalModel import NormalDistribution1D, NormalDistribution1D_unknownStd
from estimators import estimate_mean, estimate_cov
from ICNN import ICNN
from Visualizer import ICNNHeatmapVisualizer
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch

def compute_loss(model, eta_set, n_samples=256):
    """
    Compute loss of the model. Not differenciable, only used to plot loss evolution
    """
    eta = eta_set.detach().requires_grad_(True)
    A_star = model(eta).squeeze()
    theta_pred = torch.autograd.grad(A_star.sum(), eta)[0]
    theta_valid = torch.stack([theta_pred[:,0], F.softplus(theta_pred[:,1])], dim=1).detach()

    with torch.no_grad():
        stat_model = NormalDistribution1D_unknownStd(theta=theta_valid)
        t_samples = stat_model.t(stat_model.get_samples(n_samples))
        mean = estimate_mean(t_samples)
        loss = ((mean - eta_set) ** 2).sum(dim=-1).mean().item()
        if loss > 1e5:
            print("wow big loss")
    return loss

# Input
T = 4096
mu = torch.randn(T)
var = 0.01 + torch.rand(T) 
eta_set = torch.stack([mu,-(mu**2 + var)],dim=1) # expectation parameters corresponding to ~N(mu,std²)
batch_size = 64
n_sample = 512
num_epoch=16

data = TensorDataset(eta_set)
loader = DataLoader(
     dataset=data,
     batch_size=batch_size,
     shuffle=True
)

model = ICNN(2,1)
params = list(model.parameters())
visu = True # True to save a gif
train = True

eta_probe = torch.zeros((len(eta_set), eta_set.shape[1]))
eta_probe[:, 0] = eta_set[:, 0]
heatVis = ICNNHeatmapVisualizer(eta_set)

losses = []

optim = torch.optim.Adam(model.parameters(), lr=1e-2)

# Training loop
if train:
    for epoch in range(num_epoch):
        print(f"=== Epoch {epoch+1} ===")  
        for (eta_batch,) in loader:

            optim.zero_grad()

            eta_batch = eta_batch.detach().requires_grad_(True)  
            A_star = model(eta_batch).squeeze()
            theta_pred = torch.autograd.grad(
                outputs=A_star.sum(), 
                inputs=eta_batch, 
                create_graph=True
            )[0] # theta = grad_eta A*(eta)
            theta_valid = torch.stack([theta_pred[:,0],F.softplus(theta_pred[:,1])], dim=1) # ensures 1/std**2 > 0
            stat_model = NormalDistribution1D_unknownStd(theta=theta_valid)

            if (visu):
                heatVis.log(model)
                heatVis.log_grad(model)

            if torch.isnan(theta_valid).any() :
                break

            batch1 = stat_model.get_samples(n_sample)
            batch2 = stat_model.get_samples(n_sample)
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
                grad_outputs=(2*v)/batch_size,
                allow_unused=True  
            )

            grad_norm = 0
            with torch.no_grad():
                    for p, g in zip(params, grad):
                        if g is not None:
                            if g is not None and torch.isnan(g).any():
                                print("NaN détecté dans le gradient")
                            grad_norm += g.norm().item()**2
                            if p.grad is None :
                                p.grad = g
                            else :
                                p.grad+=g
            
            optim.step()

            # Compute loss
            loss = compute_loss(model, eta_set)
            losses.append(loss)

    plt.plot(losses)
    print(f"Final loss = {sum(losses[-10:])/len(losses[-10:])}")
    plt.savefig("loss.png")

if visu:
    #heatVis.save_gif()
    heatVis.save_gif_grad()
    heatVis.save_plot_GT_grad()
    heatVis.save_plot_model_grad(model)
    heatVis.save_plot_model(model)