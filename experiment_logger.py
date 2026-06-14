import json
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
    train_losses,
    test_losses,
    best_loss,
    learning_rate=None,
    note="",
    filename="experiments.jsonl",
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
        "final_train_loss": sum(train_losses[-10:]) / min(len(train_losses), 10) if train_losses else None,
        "final_test_loss": sum(test_losses[-10:]) / min(len(test_losses), 10) if test_losses else None,
        "best_loss": best_loss,
    }

    with open(filename, "a") as f:
        f.write(json.dumps(results) + "\n")

    print(f"Results successfully saved in {filename}")