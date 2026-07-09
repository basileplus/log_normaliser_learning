from Sampler import NormalDistribution1D_etaParam
from estimators import estimate_mean, estimate_cov
from ICNN import ICNN
from Visualizer import ICNN1DVisualizer, save_loss_plot
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from experiment_logger import logExperimentResult
import torch

def compute_loss(model, theta_set, n_samples=256):
    """
    Compute loss of the model. Not differenciable, only used to plot loss evolution
    """
    theta = theta_set.detach().requires_grad_(True)
    A = model(theta).squeeze()
    eta_pred = torch.autograd.grad(A.sum(), theta)[0].detach()

    with torch.no_grad():
        stat_model = NormalDistribution1D_etaParam(eta=eta_pred)
        t_samples = stat_model.t(stat_model.get_samples(n_samples))
        mean = estimate_mean(t_samples)
        loss = ((mean - eta_pred) ** 2).sum(dim=-1).mean().item()
        if loss > 1e5:
            print("wow big loss")
    return loss

# Input
T = 4096
mu = torch.randn(T)
var = torch.ones_like(mu)
theta_set = (mu/var).unsqueeze(1)
batch_size = 16
n_sample = 512
num_epoch=4

perm = torch.randperm(T)
theta_train_set = theta_set[perm[:int(0.8*T)]]
theta_test_set = theta_set[perm[int(0.8*T):]]

train_data = TensorDataset(theta_train_set)
loader = DataLoader(
     dataset=train_data,
     batch_size=batch_size,
     shuffle=True
)

model = ICNN(1,1)
params = list(model.parameters())
visu = True # True to save a gif
train = True

heatVis = ICNN1DVisualizer(theta_set)

train_losses = []
test_losses = []
batch_idx=0
log_every = int((T*num_epoch)/(50*batch_size))
best_loss = float("inf")

optim = torch.optim.Adam(model.parameters(), lr=1e-2)

# Training loop
if train:
    for epoch in range(num_epoch):
        print(f"=== Epoch {epoch+1} | train_loss={sum(train_losses[-10:])/len(train_losses[-10:]) if train_losses else "N/A"} | test_loss={sum(test_losses[-10:])/len(test_losses[-10:]) if test_losses else "N/A"}===")        
        for (theta_batch,) in loader:

            optim.zero_grad()

            theta_batch = theta_batch.detach().requires_grad_(True)  
            A = model(theta_batch).squeeze()
            eta_pred = torch.autograd.grad(
                outputs=A.sum(), 
                inputs=theta_batch, 
                create_graph=True
            )[0] # theta = grad_eta A*(eta)
            stat_model = NormalDistribution1D_etaParam(eta=eta_pred)

            if (visu):
                if batch_idx % log_every ==0:
                    heatVis.log(model)
                    heatVis.log_grad(model)

            if torch.isnan(eta_pred).any() :
                break

            batch1 = stat_model.get_samples(n_sample)
            batch2 = stat_model.get_samples(n_sample)
            t1 = stat_model.t(batch1)
            t2 = stat_model.t(batch2)
            mean = estimate_mean(t1)

            v = (mean - eta_pred).unsqueeze(-1).squeeze(-1)

            # Clamp v, prevent gradient explosions occuring from cov estimate (var X^2~sigma^4)
            v_norm = v.norm()
            if v_norm > 10:
                v = 10 * v / v_norm

            grad = torch.autograd.grad(
                outputs=eta_pred,
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
            train_loss = compute_loss(model, theta_train_set)
            test_loss = compute_loss(model, theta_test_set)
            if test_loss < best_loss:  
                best_loss = test_loss
                torch.save(model.state_dict(), "best_model.pt")

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            batch_idx+=1

    print(f"Final training loss = {sum(train_losses[-10:])/len(train_losses[-10:])} | Final test loss = {sum(test_losses[-10:])/len(test_losses[-10:])}")
    print(f"Best loss = {best_loss}")

    # load best performing model
    model.load_state_dict(torch.load("best_model.pt"))
    if (visu):
        heatVis.log(model)
        heatVis.log_grad(model)



# Log results in a csv
if train:
    exp_id = logExperimentResult(
        optimizer=optim,
        target_distrib="1D Gaussian known std, eta param",
        mu=mu,
        var=var,
        training_set="torch.randn(T)",
        batch_size=batch_size,
        dataset_size=T,
        n_epochs=num_epoch,
        n_samples=n_sample,
        train_losses = train_losses,
        test_losses = test_losses,
        best_loss=best_loss,
        note="Dual param, Gaussian, known std",
    )

if visu:
    if train:
        heatVis.save_gif(f"visualizations/knownStd_dual_{exp_id}_model.gif")
        heatVis.save_gif_grad(f"visualizations/knownStd_dual_{exp_id}_grad.gif")
        heatVis.save_plot_GT_grad(f"visualizations/knownStd_dual_{exp_id}_gt_grad.png")
        heatVis.save_plot_model_grad(model, f"visualizations/knownStd_dual_{exp_id}_model_grad.png")
        heatVis.save_plot_model(model, f"visualizations/knownStd_dual_{exp_id}_model.png")
        save_loss_plot(train_losses, test_losses, filename=f"visualizations/knownStd_dual_{exp_id}_loss.png")
    else :
        heatVis.save_plot_GT_grad()
