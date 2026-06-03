import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io

class DensityVisualizer:
    """
    Used to create a gif of the learned density histogram evolving through training.
    1. Log some samples with .log(function) during training
    2. call .save_fig() at the end
    """
    def __init__(self):
        self.snapshots = []

    def log(self, stat_model, n_samples=5000):
        with torch.no_grad():
            samples = stat_model.get_samples(n_samples).cpu()
        if samples.ndim ==1:
            self.snapshots.append(samples)
        else :
            self.snapshots.append(samples[:,0])

    def save_gif(
        self,
        filename="density_evolution.gif",
        bins=100,
        duration=0.1,
        xlim=None,
        ylim=None
    ):
        frames = []
        all_samples = torch.cat(self.snapshots)
        xmin = all_samples.min().item()
        xmax = all_samples.max().item()
        for i, samples in enumerate(self.snapshots):
            fig, ax = plt.subplots()
            ax.hist(
                samples.numpy(),
                bins=bins,
                density=True
            )
            ax.set_xlim(xmin, xmax)
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_title(f"Snapshot {i}")
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.imread(buf))
        imageio.mimsave(filename, frames, duration=duration)
        print(f"File {filename} succesfully saved")