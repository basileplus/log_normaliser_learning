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

class ICNNHeatmapVisualizer:
    """
    Saves a gif of the heatmap of ICNN(eta) over a 2D grid during training.
    The grid is built automatically from the data range passed at init.
    1. Init with eta_set to determine the grid extent
    2. Call .log(model) during training to snapshot A*(eta) over the grid
    3. Call .save_gif() at the end
 
    Uses quantiles to clip the range so outliers in eta don't collapse the heatmap.
    Color scale is fixed across all frames so evolution is visible.
    """
    def __init__(self, eta_set, resolution=240, quantile=0.05):
        self.eta1_min = torch.quantile(eta_set[:, 0], quantile).item()
        self.eta1_max = torch.quantile(eta_set[:, 0], 1 - quantile).item()
        self.eta2_min = torch.quantile(eta_set[:, 1], quantile).item()
        self.eta2_max = torch.quantile(eta_set[:, 1], 1 - quantile).item()
 
        eta1_range = torch.linspace(self.eta1_min, self.eta1_max, resolution)
        eta2_range = torch.linspace(self.eta2_min, self.eta2_max, resolution)
        eta1_grid, eta2_grid = torch.meshgrid(eta1_range, eta2_range, indexing='ij')
 
        # (resolution^2, 2) — flat grid ready for model input
        self.eta_grid = torch.stack([eta1_grid.flatten(), eta2_grid.flatten()], dim=1)
        self.resolution = resolution
        self.eta1_range = eta1_range
        self.eta2_range = eta2_range
        self.snapshots = []
 
    def log(self, model):
        with torch.no_grad():
            values = model(self.eta_grid).squeeze()          # (resolution^2,)
        self.snapshots.append(values.reshape(self.resolution, self.resolution).cpu())
 
    def save_gif(self, filename="icnn_heatmap.gif", duration=0.1):
        # Fix colour scale across all frames so changes are visible
        vmin = min(s.min().item() for s in self.snapshots)
        vmax = max(s.max().item() for s in self.snapshots)
 
        extent = [
            self.eta1_range[0].item(), self.eta1_range[-1].item(),
            self.eta2_range[0].item(), self.eta2_range[-1].item(),
        ]
 
        frames = []
        for i, snapshot in enumerate(self.snapshots):
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(
                snapshot.T,         
                extent=extent,
                origin='lower',
                aspect='auto',
                vmin=vmin, vmax=vmax,
                cmap='viridis',
            )
            ax.contour(snapshot.T, extent=extent, origin='lower', levels=10, colors='white', linewidths=0.5, alpha=0.5)
            plt.colorbar(im, ax=ax, label="$A^*(\\eta)$")
            ax.set_title(f"$A^*(\\eta)$ — step {i}")
            ax.set_xlabel("$\\eta_1$")
            ax.set_ylabel("$\\eta_2$")
 
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.imread(buf))
 
        imageio.mimsave(filename, frames, duration=duration)
        print(f"File {filename} successfully saved")

    def save_plot_GT(self, filename="ground_truth_heatmap.png"):
        eta1 = self.eta1_range
        eta2 = self.eta2_range
        eta1_grid, eta2_grid = torch.meshgrid(eta1, eta2, indexing='ij')

        A = -eta1_grid**2 / (4 * eta2_grid) - 0.5 * torch.log(-2 * eta2_grid)

        plt.figure(figsize=(6, 5))
        im = plt.imshow(A.T, extent=[self.eta1_min, self.eta1_max, self.eta2_min, self.eta2_max],origin='lower', aspect='auto', cmap='viridis')
        plt.contour(A.T, extent=[self.eta1_min, self.eta1_max, self.eta2_min, self.eta2_max],origin='lower', levels=10, colors='white', linewidths=0.5, alpha=0.5)
        plt.colorbar(im, label="$A^*(\\eta)$")
        plt.title("Ground truth $-\\frac{\\eta_1^2}{4\\eta_2} - \\frac{1}{2}\\log(-2\\eta_2)$")
        plt.xlabel("$\\eta_1$"); plt.ylabel("$\\eta_2$")
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        print(f"File {filename} successfully saved")
        plt.close()
    
    def save_plot_model(self, model, filename="icnn_heatmap.png"):
        with torch.no_grad():
            values = model(self.eta_grid).squeeze()
        snapshot = values.reshape(self.resolution, self.resolution).cpu()

        extent = [self.eta1_min, self.eta1_max, self.eta2_min, self.eta2_max]

        plt.figure(figsize=(6, 5))
        im = plt.imshow(snapshot.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')
        plt.contour(snapshot.T, extent=extent, origin='lower', levels=10, colors='white', linewidths=0.5, alpha=0.5)
        plt.colorbar(im, label="$A^*(\\eta)$")
        plt.title("Learned $A^*(\\eta)$")
        plt.xlabel("$\\eta_1$"); plt.ylabel("$\\eta_2$")
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        print(f"File {filename} successfully saved")
        plt.close()