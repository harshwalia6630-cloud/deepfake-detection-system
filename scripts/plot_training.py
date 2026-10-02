"""Plot accuracy / AUC / loss curves from reports/training_log.csv."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(head_epochs=4):
    log = pd.read_csv(ROOT / "reports/training_log.csv")
    epochs = log["epoch"] + 1
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ("acc", "auc", "loss")):
        ax.plot(epochs, log[metric], label="train")
        ax.plot(epochs, log[f"val_{metric}"], label="validation")
        ax.axvline(head_epochs + 0.5, color="grey", ls="--", lw=0.8)
        ax.set(title=metric.upper() if metric != "acc" else "Accuracy", xlabel="epoch")
        ax.legend()
    axes[0].text(head_epochs + 0.7, axes[0].get_ylim()[0] + 0.02, "fine-tuning starts", color="grey", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "reports/training_curves.png", dpi=120)


if __name__ == "__main__":
    main()
