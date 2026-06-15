import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io
import torch.nn.functional as F
import numpy as np

def save_loss_plot(train_losses, test_losses, filename="loss.png"):
    plt.figure()
    plt.plot(train_losses, label="Train loss")
    plt.plot(test_losses, label="Test loss")
    plt.legend()
    plt.savefig(filename)

class ICNN2DHeatmapVisualizer:
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
        self.grad_snapshots = []
 
    def log(self, model):
        with torch.no_grad():
            values = model(self.eta_grid).squeeze()          # (resolution^2,)
        self.snapshots.append(values.reshape(self.resolution, self.resolution).cpu())

    def log_grad(self, model):
        """
        Snapshot theta_valid = (grad_eta ICNN(eta)[0], softplus(grad_eta ICNN(eta)[1]))
        over the full grid. Call this alongside log() during training.
        """
        eta_grid = self.eta_grid.detach().requires_grad_(True)
        # sum over the batch to get a scalar, then differentiate w.r.t. eta_grid
        A_star = model(eta_grid).squeeze().sum()
        theta_pred = torch.autograd.grad(A_star, eta_grid)[0]  # (resolution^2, 2)
        theta_valid = torch.stack([
            theta_pred[:, 0].detach(),
            F.softplus(theta_pred[:, 1]).detach(),
        ], dim=1)  # (resolution^2, 2)
        # store as (resolution, resolution, 2)
        self.grad_snapshots.append(
            theta_valid.reshape(self.resolution, self.resolution, 2).cpu()
        )
 
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

    def save_gif_grad(self, filename="icnn_grad_heatmap.gif", duration=0.1):
        """
        GIF of the learned gradient theta_valid = (theta_1, softplus(theta_2))
        during training. Each frame is a two-panel figure (one panel per component).
        Requires prior calls to log_grad(model) during training.
        """
        if not self.grad_snapshots:
            print("No gradient snapshots found. Call log_grad(model) during training first.")
            return
 
        # Fixed, clipped colour scale per component across all frames
        all_t1 = torch.cat([s[:, :, 0].flatten() for s in self.grad_snapshots])
        all_t2 = torch.cat([s[:, :, 1].flatten() for s in self.grad_snapshots])
        vmin1 = torch.nanquantile(all_t1, 0.02).item()
        vmax1 = torch.nanquantile(all_t1, 0.98).item()
        vmin2 = torch.nanquantile(all_t2, 0.02).item()
        vmax2 = torch.nanquantile(all_t2, 0.98).item()
 
        extent = [self.eta1_min, self.eta1_max, self.eta2_min, self.eta2_max]
 
        frames = []
        for i, snap in enumerate(self.grad_snapshots):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
            # theta_1: diverging colormap (can be negative)
            im1 = axes[0].imshow(
                snap[:, :, 0].T, extent=extent, origin='lower', aspect='auto',
                vmin=vmin1, vmax=vmax1, cmap='RdBu_r',
            )
            plt.colorbar(im1, ax=axes[0], label=r"$\hat\theta_1(\eta)$")
            axes[0].set_title(r"$\hat\theta_1 = \partial_{\eta_1} ICNN$" + f"  —  step {i}")
            axes[0].set_xlabel(r"$\eta_1$"); axes[0].set_ylabel(r"$\eta_2$")
 
            # theta_2: sequential colormap (always positive)
            im2 = axes[1].imshow(
                snap[:, :, 1].T, extent=extent, origin='lower', aspect='auto',
                vmin=vmin2, vmax=vmax2, cmap='plasma',
            )
            plt.colorbar(im2, ax=axes[1], label=r"$\hat\theta_2(\eta)$")
            axes[1].set_title(r"$\hat\theta_2 = \mathrm{softplus}(\partial_{\eta_2} ICNN)$" + f"  —  step {i}")
            axes[1].set_xlabel(r"$\eta_1$"); axes[1].set_ylabel(r"$\eta_2$")
 
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.imread(buf))
 
        imageio.mimsave(filename, frames, duration=duration)
        print(f"File {filename} successfully saved")

    def save_plot_GT_grad(self, filename="ground_truth_grad_heatmap.png"):
        """
        Two-panel static plot of the ground truth gradient of A*:
          theta_1*(eta) = eta_1 / sigma^2
          theta_2*(eta) = 1 / (2*sigma^2)
        where sigma^2 = -eta_1^2 - eta_2.
        Points outside the valid domain (sigma^2 <= 0) are shown as NaN (white).
        Colour scale clipped to 2%–98% quantiles over valid points.
        """
        eta1_grid, eta2_grid = torch.meshgrid(self.eta1_range, self.eta2_range, indexing='ij')
        sigma2 = -(eta1_grid**2 + eta2_grid)   # (res, res)
 
        valid = sigma2 > 0
        theta1_gt = torch.full_like(sigma2, float('nan'))
        theta2_gt = torch.full_like(sigma2, float('nan'))
        theta1_gt[valid] = eta1_grid[valid] / sigma2[valid]
        theta2_gt[valid] = 0.5 / sigma2[valid]
 
        # Clip colour scale to avoid boundary blow-up dominating the colourbar
        t1_vals = theta1_gt[valid]
        t2_vals = theta2_gt[valid]
        vmin1 = torch.quantile(t1_vals, 0.02).item()
        vmax1 = torch.quantile(t1_vals, 0.98).item()
        vmin2 = torch.quantile(t2_vals, 0.02).item()
        vmax2 = torch.quantile(t2_vals, 0.98).item()
 
        extent = [self.eta1_min, self.eta1_max, self.eta2_min, self.eta2_max]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
        im1 = axes[0].imshow(
            theta1_gt.T.numpy(), extent=extent, origin='lower', aspect='auto',
            vmin=vmin1, vmax=vmax1, cmap='RdBu_r',
        )
        plt.colorbar(im1, ax=axes[0], label=r"$\theta_1^*(\eta)$")
        axes[0].set_title(r"GT $\;\theta_1^* = \eta_1 / \sigma^2$")
        axes[0].set_xlabel(r"$\eta_1$"); axes[0].set_ylabel(r"$\eta_2$")
 
        im2 = axes[1].imshow(
            theta2_gt.T.numpy(), extent=extent, origin='lower', aspect='auto',
            vmin=vmin2, vmax=vmax2, cmap='plasma',
        )
        plt.colorbar(im2, ax=axes[1], label=r"$\theta_2^*(\eta)$")
        axes[1].set_title(r"GT $\;\theta_2^* = 1 / (2\sigma^2)$")
        axes[1].set_xlabel(r"$\eta_1$"); axes[1].set_ylabel(r"$\eta_2$")
 
        plt.tight_layout()
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        print(f"File {filename} successfully saved")
        plt.close()
  
    def save_plot_model_grad(self, model, filename="icnn_grad_heatmap.png"):
        """
        Two-panel static plot of the model's predicted gradient
        theta_valid = (theta_1, softplus(theta_2)) at the end of training.
        """
        eta_grid = self.eta_grid.detach().requires_grad_(True)
        A_star = model(eta_grid).squeeze().sum()
        theta_pred = torch.autograd.grad(A_star, eta_grid)[0]
        theta_valid = torch.stack([
            theta_pred[:, 0].detach(),
            F.softplus(theta_pred[:, 1]).detach(),
        ], dim=1).reshape(self.resolution, self.resolution, 2).cpu()
 
        # Use the same clipped colour range as the GT plot for visual alignment
        eta1_grid, eta2_grid = torch.meshgrid(self.eta1_range, self.eta2_range, indexing='ij')
        sigma2 = -(eta1_grid**2 + eta2_grid)
        valid = sigma2 > 0
        theta1_gt = torch.where(valid, eta1_grid / sigma2, torch.zeros_like(sigma2))
        theta2_gt = torch.where(valid, 0.5 / sigma2, torch.zeros_like(sigma2))
        vmin1 = torch.quantile(theta1_gt[valid], 0.02).item()
        vmax1 = torch.quantile(theta1_gt[valid], 0.98).item()
        vmin2 = torch.quantile(theta2_gt[valid], 0.02).item()
        vmax2 = torch.quantile(theta2_gt[valid], 0.98).item()
 
        extent = [self.eta1_min, self.eta1_max, self.eta2_min, self.eta2_max]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
        im1 = axes[0].imshow(
            theta_valid[:, :, 0].T.numpy(), extent=extent, origin='lower', aspect='auto',
            vmin=vmin1, vmax=vmax1, cmap='RdBu_r',
        )
        plt.colorbar(im1, ax=axes[0], label=r"$\hat\theta_1(\eta)$")
        axes[0].set_title(r"Model $\hat\theta_1 = \partial_{\eta_1} ICNN$")
        axes[0].set_xlabel(r"$\eta_1$"); axes[0].set_ylabel(r"$\eta_2$")
 
        im2 = axes[1].imshow(
            theta_valid[:, :, 1].T.numpy(), extent=extent, origin='lower', aspect='auto',
            vmin=vmin2, vmax=vmax2, cmap='plasma',
        )
        plt.colorbar(im2, ax=axes[1], label=r"$\hat\theta_2(\eta)$")
        axes[1].set_title(r"Model $\hat\theta_2 = \mathrm{softplus}(\partial_{\eta_2} ICNN)$")
        axes[1].set_xlabel(r"$\eta_1$"); axes[1].set_ylabel(r"$\eta_2$")
 
        plt.tight_layout()
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        print(f"File {filename} successfully saved")
        plt.close()

    def save_plot_GT(self, filename="ground_truth_heatmap.png"):
        eta1 = self.eta1_range
        eta2 = self.eta2_range
        eta1_grid, eta2_grid = torch.meshgrid(eta1, eta2, indexing='ij')

        A = -0.5 *(1+ torch.log(-(eta1_grid**2 + eta2_grid)))

        plt.figure(figsize=(6, 5))
        im = plt.imshow(A.T, extent=[self.eta1_min, self.eta1_max, self.eta2_min, self.eta2_max],origin='lower', aspect='auto', cmap='viridis')
        plt.contour(A.T, extent=[self.eta1_min, self.eta1_max, self.eta2_min, self.eta2_max],origin='lower', levels=10, colors='white', linewidths=0.5, alpha=0.5)
        plt.colorbar(im, label="$A^*(\\eta)$")
        plt.title("Ground truth $ - \\frac{1}{2}(1+\\log(-(\\eta_1^2+\\eta_2))$")
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
        plt.title("Learned $ICNN(\\eta)$")
        plt.xlabel("$\\eta_1$"); plt.ylabel("$\\eta_2$")
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        print(f"File {filename} successfully saved")
        plt.close()


class ICNN1DVisualizer:
    def __init__(self, eta_set, resolution=400, quantile=0.02):
        self.eta_min = torch.quantile(eta_set[:, 0], quantile).item()
        self.eta_max = torch.quantile(eta_set[:, 0], 1 - quantile).item()

        self.eta_grid = torch.linspace(
            self.eta_min,
            self.eta_max,
            resolution,
        ).unsqueeze(1)

        self.snapshots = []
        self.grad_snapshots = []

    def log(self, model):
        with torch.no_grad():
            values = model(self.eta_grid).squeeze().cpu()
        self.snapshots.append(values)

    def log_grad(self, model):
        eta = self.eta_grid.detach().requires_grad_(True)
        A = model(eta).sum()
        theta = torch.autograd.grad(A, eta)[0].detach().squeeze().cpu()
        self.grad_snapshots.append(theta)

    def save_gif(self, filename="icnn1d.gif", duration=0.1):
        ymin = min(s.min().item() for s in self.snapshots)
        ymax = max(s.max().item() for s in self.snapshots)

        frames = []

        for i, snap in enumerate(self.snapshots):
            fig, ax = plt.subplots()

            ax.plot(
                self.eta_grid.squeeze().numpy(),
                snap.numpy(),
            )

            ax.set_ylim(ymin, ymax)
            ax.set_xlabel(r"$\eta$")
            ax.set_ylabel(r"$A^*(\eta)$")
            ax.set_title(f"Step {i}")

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.imread(buf))

        imageio.mimsave(filename, frames, duration=duration)
        print(f"File {filename} successfully saved")

    def save_gif_grad(self, filename="icnn1d_grad.gif", duration=0.1):
        ymin = min(s.min().item() for s in self.grad_snapshots)
        ymax = max(s.max().item() for s in self.grad_snapshots)

        frames = []

        for i, snap in enumerate(self.grad_snapshots):
            fig, ax = plt.subplots()

            ax.plot(
                self.eta_grid.squeeze().numpy(),
                snap.numpy(),
            )

            ax.set_ylim(ymin, ymax)
            ax.set_xlabel(r"$\eta$")
            ax.set_ylabel(r"$\theta(\eta)$")
            ax.set_title(f"Step {i}")

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.imread(buf))

        imageio.mimsave(filename, frames, duration=duration)
        print(f"File {filename} successfully saved")

    def save_plot_GT(self, filename="ground_truth.png"):
        eta = self.eta_grid.squeeze()

        # Gaussian with known variance = 1:
        # A*(eta) = 1/2 eta^2
        A = 0.5 * eta ** 2

        plt.figure(figsize=(6, 4))
        plt.plot(eta.numpy(), A.numpy())
        plt.xlabel(r"$\eta$")
        plt.ylabel(r"$A^*(\eta)$")
        plt.title("Ground truth")
        plt.savefig(filename, dpi=120)
        plt.close()
        print(f"File {filename} successfully saved")

    def save_plot_GT_grad(self, filename="ground_truth_grad.png"):
        eta = self.eta_grid.squeeze()

        # theta = eta
        plt.figure(figsize=(6, 4))
        plt.plot(eta.numpy(), eta.numpy())
        plt.xlabel(r"$\eta$")
        plt.ylabel(r"$\theta(\eta)$")
        plt.title("Ground truth gradient")
        plt.savefig(filename, dpi=120)
        plt.close()
        print(f"File {filename} successfully saved")

    def save_plot_model(self, model, filename="model.png"):
        with torch.no_grad():
            values = model(self.eta_grid).squeeze()

        plt.figure(figsize=(6, 4))
        plt.plot(
            self.eta_grid.squeeze().numpy(),
            values.cpu().numpy(),
        )
        plt.xlabel(r"$\eta$")
        plt.ylabel(r"$A^*(\eta)$")
        plt.title("Learned ICNN")
        plt.savefig(filename, dpi=120)
        plt.close()
        print(f"File {filename} successfully saved")

    def save_plot_model_grad(self, model, filename="model_grad.png"):
        eta = self.eta_grid.detach().requires_grad_(True)

        A = model(eta).sum()
        theta = torch.autograd.grad(A, eta)[0].detach().squeeze()

        plt.figure(figsize=(6, 4))
        plt.plot(
            eta.detach().squeeze().numpy(),
            theta.cpu().numpy(),
        )
        plt.xlabel(r"$\eta$")
        plt.ylabel(r"$\theta(\eta)$")
        plt.title("Learned gradient")
        plt.savefig(filename, dpi=120)
        plt.close()
        print(f"File {filename} successfully saved")

class UniformVisualizer:
    """
    Saves a gif of the heatmap of ICNN(eta) over a 2D grid during training.
    1. Init with eta_set to determine the grid extent
    2. Call .log(model) during training to snapshot A*(eta) over the grid
    3. Call .save_gif() at the end
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
        self.grad_snapshots = []
 
    def log(self, model):
        with torch.no_grad():
            values = model(self.eta_grid).squeeze()          # (resolution^2,)
        self.snapshots.append(values.reshape(self.resolution, self.resolution).cpu())

    def log_grad(self, model):
        """
        Snapshot theta_valid = (grad_eta ICNN(eta)[0], softplus(grad_eta ICNN(eta)[1]))
        over the full grid. Call this alongside log() during training.
        """
        eta_grid = self.eta_grid.detach().requires_grad_(True)
        # sum over the batch to get a scalar, then differentiate w.r.t. eta_grid
        A_star = model(eta_grid).squeeze().sum()
        theta_pred = torch.autograd.grad(A_star, eta_grid)[0]  # (resolution^2, 2)
        theta_valid = torch.stack([
            theta_pred[:, 0].detach(),
            F.softplus(theta_pred[:, 1]).detach(),
        ], dim=1)  # (resolution^2, 2)
        # store as (resolution, resolution, 2)
        self.grad_snapshots.append(
            theta_valid.reshape(self.resolution, self.resolution, 2).cpu()
        )
 
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

    def save_gif_grad(self, filename="icnn_grad_heatmap.gif", duration=0.1):
        """
        GIF of the learned gradient theta_valid = (theta_1, softplus(theta_2))
        during training. Each frame is a two-panel figure (one panel per component).
        Requires prior calls to log_grad(model) during training.
        """
        if not self.grad_snapshots:
            print("No gradient snapshots found. Call log_grad(model) during training first.")
            return
 
        # Fixed, clipped colour scale per component across all frames
        all_t1 = torch.cat([s[:, :, 0].flatten() for s in self.grad_snapshots])
        all_t2 = torch.cat([s[:, :, 1].flatten() for s in self.grad_snapshots])
        vmin1 = torch.nanquantile(all_t1, 0.02).item()
        vmax1 = torch.nanquantile(all_t1, 0.98).item()
        vmin2 = torch.nanquantile(all_t2, 0.02).item()
        vmax2 = torch.nanquantile(all_t2, 0.98).item()
 
        extent = [self.eta1_min, self.eta1_max, self.eta2_min, self.eta2_max]
 
        frames = []
        for i, snap in enumerate(self.grad_snapshots):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
            # theta_1: diverging colormap (can be negative)
            im1 = axes[0].imshow(
                snap[:, :, 0].T, extent=extent, origin='lower', aspect='auto',
                vmin=vmin1, vmax=vmax1, cmap='RdBu_r',
            )
            plt.colorbar(im1, ax=axes[0], label=r"$\hat\theta_1(\eta)$")
            axes[0].set_title(r"$\hat\theta_1 = \partial_{\eta_1} ICNN$" + f"  —  step {i}")
            axes[0].set_xlabel(r"$\eta_1$"); axes[0].set_ylabel(r"$\eta_2$")
 
            # theta_2: sequential colormap (always positive)
            im2 = axes[1].imshow(
                snap[:, :, 1].T, extent=extent, origin='lower', aspect='auto',
                vmin=vmin2, vmax=vmax2, cmap='plasma',
            )
            plt.colorbar(im2, ax=axes[1], label=r"$\hat\theta_2(\eta)$")
            axes[1].set_title(r"$\hat\theta_2 = \mathrm{softplus}(\partial_{\eta_2} ICNN)$" + f"  —  step {i}")
            axes[1].set_xlabel(r"$\eta_1$"); axes[1].set_ylabel(r"$\eta_2$")
 
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.imread(buf))
 
        imageio.mimsave(filename, frames, duration=duration)
        print(f"File {filename} successfully saved")
  
    def save_plot_model_grad(self, model, filename="icnn_grad_heatmap.png"):
        """
        Two-panel static plot of the model's predicted gradient
        """
        eta_grid = self.eta_grid.detach().requires_grad_(True)
        A_star = model(eta_grid).squeeze().sum()
        theta_pred = torch.autograd.grad(A_star, eta_grid)[0]
        theta_valid = theta_pred
        # Use the same clipped colour range as the GT plot for visual alignment
        eta1_grid, eta2_grid = torch.meshgrid(self.eta1_range, self.eta2_range, indexing='ij')
        sigma2 = -(eta1_grid**2 + eta2_grid)
 
        extent = [self.eta1_min, self.eta1_max, self.eta2_min, self.eta2_max]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
        im1 = axes[0].imshow(
            theta_valid[:, :, 0].T.numpy(), extent=extent, origin='lower', aspect='auto', cmap='RdBu_r',
        )
        plt.colorbar(im1, ax=axes[0], label=r"$\hat\theta_1(\eta)$")
        axes[0].set_title(r"Model $\hat\theta_1 = \partial_{\eta_1} ICNN$")
        axes[0].set_xlabel(r"$\eta_1$"); axes[0].set_ylabel(r"$\eta_2$")
 
        im2 = axes[1].imshow(
            theta_valid[:, :, 1].T.numpy(), extent=extent, origin='lower', aspect='auto', cmap='plasma',
        )
        plt.colorbar(im2, ax=axes[1], label=r"$\hat\theta_2(\eta)$")
        axes[1].set_title(r"Model $\hat\theta_2 = \mathrm{softplus}(\partial_{\eta_2} ICNN)$")
        axes[1].set_xlabel(r"$\eta_1$"); axes[1].set_ylabel(r"$\eta_2$")
 
        plt.tight_layout()
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
        plt.title("Learned $ICNN(\\eta)$")
        plt.xlabel("$\\eta_1$"); plt.ylabel("$\\eta_2$")
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        print(f"File {filename} successfully saved")
        plt.close()