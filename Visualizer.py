import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io

class ConjugacyVisualizer:
    """
    Used to create a gif of the learned mapping through training. Visualize first coordinate theta[0]
    1. Init with a set of eta on which we will plot the mapping
    2. call .log() during training to save a snapshot of the conjugation map
    3. call save_gif at the end
    """
    def __init__(self, eta_probe):
        self.eta_probe = eta_probe  
        self.snapshots = []


    def log(self, model):
        thetas = []
        for eta in self.eta_probe:
            eta = eta.clone().requires_grad_(True)
            A_star = model(eta)
            theta = torch.autograd.grad(
                outputs = A_star,
                inputs=eta
            )[0]
            thetas.append(theta.detach().cpu())
        self.snapshots.append(torch.stack(thetas))

    def save_gif(
        self,
        filename="conjugacy.gif",
        duration=0.1,
    ):
        frames = []
        eta1 = self.eta_probe[:, 0].cpu()
        all_theta1 = torch.cat([snap[:, 0] for snap in self.snapshots])
        for i, theta_snapshot in enumerate(self.snapshots):
            fig, ax = plt.subplots()
            ax.scatter(
                eta1,
                theta_snapshot[:,0]
            )   
            ax.set_title(f"Snapshot {i}")
            ax.set_xlabel("$\eta_1$")
            ax.set_ylabel("$\\theta_1$")
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.imread(buf))
        imageio.mimsave(filename, frames, duration=duration)
        print(f"File {filename} succesfully saved")