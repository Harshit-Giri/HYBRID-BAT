
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "nasa_dataset.npz"
PLOT_DIR = ROOT / "results" / "plots"


def main():
    data = np.load(DATA_PATH, allow_pickle=False)

    X = data["X"]
    y = data["y"]
    cell_ids = data["cell_ids"]
    cycle_indices = data["cycle_indices"]

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: SOH against charge-cycle index for each cell
    plt.figure(figsize=(10, 5))

    for cell in np.unique(cell_ids):
        mask = cell_ids == cell
        order = np.argsort(cycle_indices[mask])

        plt.plot(
            cycle_indices[mask][order],
            y[mask][order],
            marker=".",
            markersize=3,
            label=cell,
        )

    plt.xlabel("Charge cycle index")
    plt.ylabel("SOH (fraction)")
    plt.title("NASA Battery SOH vs Cycle")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "soh_vs_cycle.png", dpi=150)
    plt.close()

    # Plot 2: Mean temperature over the first 100 points
    plt.figure(figsize=(10, 5))

    for cell in np.unique(cell_ids):
        mask = cell_ids == cell
        order = np.argsort(cycle_indices[mask])

        mean_temp = X[mask, 1, :].mean(axis=1)

        plt.plot(
            cycle_indices[mask][order],
            mean_temp[order],
            marker=".",
            markersize=3,
            label=cell,
        )

    plt.xlabel("Charge cycle index")
    plt.ylabel("Mean temperature (°C)")
    plt.title("NASA Battery Mean Charge Temperature vs Cycle")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "temperature_vs_cycle.png", dpi=150)
    plt.close()

    print("Plots saved:")
    print(PLOT_DIR / "soh_vs_cycle.png")
    print(PLOT_DIR / "temperature_vs_cycle.png")


if __name__ == "__main__":
    main()

