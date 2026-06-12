# experiment_logger.py

import csv
import os
from datetime import datetime


def logExperimentResult(
    *,
    optimizer,
    mu,
    var,
    batch_size,
    dataset_size,
    n_epochs,
    n_samples,
    losses,
    learning_rate=None,
    note="",
    filename="experiments.csv",
):
    results = {
        "datetime": datetime.now().isoformat(),
        "note": note,
        "optimizer": type(optimizer).__name__,
        "learning_rate": (
            learning_rate
            if learning_rate is not None
            else optimizer.param_groups[0]["lr"]
        ),
        "batch_size": batch_size,
        "dataset_size": dataset_size,
        "mu_mean": mu.mean().item(),
        "mu_std": mu.std().item(),
        "var_mean": var.mean().item(),
        "var_std": var.std().item(),
        "n_epochs": n_epochs,
        "n_samples": n_samples,
        "final_loss": sum(losses[-10:]) / min(len(losses), 10),
    }

    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(results)