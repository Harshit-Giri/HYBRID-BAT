
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "nasa_dataset.npz"
PLOT_DIR = ROOT / "results" / "plots"


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}\n"
            "Run: python src/build_dataset.py"
        )

    data = np.load(DATA_PATH, allow_pickle=False)

    X = data["X"]
    y = data["y"]
    cell_ids = data["cell_ids"]
    cycle_indices = data["cycle_indices"]

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("HYBRID-BAT | NASA DATASET DIAGNOSTICS")
    print("=" * 55)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Cells: {np.unique(cell_ids)}")

    # --------------------------------------------------
    # 1. DATA VALIDATION
    # --------------------------------------------------

    print("\n1. BASIC DATA VALIDATION")

    print(f"NaN values in X: {np.isnan(X).sum()}")
    print(f"NaN values in y: {np.isnan(y).sum()}")
    print(f"Infinite values in X: {np.isinf(X).sum()}")
    print(f"Infinite values in y: {np.isinf(y).sum()}")

    print(
        f"SOH range: {y.min():.4f} to {y.max():.4f}"
    )

    # --------------------------------------------------
    # 2. TEMPERATURE DIAGNOSTICS
    # --------------------------------------------------

    print("\n2. TEMPERATURE DIAGNOSTICS")
    print(
        "Cell     Min (C)    Max (C)    Mean (C)    Std (C)"
    )
    print("-" * 58)

    temperature_means = {}

    for cell in np.unique(cell_ids):

        mask = cell_ids == cell

        # X[:, 1, :] contains temperature readings
        # for the first 100 points of each charge sample.
        temperatures = X[mask, 1, :]

        temp_min = float(temperatures.min())
        temp_max = float(temperatures.max())
        temp_mean = float(temperatures.mean())
        temp_std = float(temperatures.std())

        temperature_means[cell] = temperatures.mean(axis=1)

        print(
            f"{cell:<8} "
            f"{temp_min:>8.2f} "
            f"{temp_max:>10.2f} "
            f"{temp_mean:>11.2f} "
            f"{temp_std:>10.2f}"
        )

    # --------------------------------------------------
    # 3. SOH VS CYCLE
    # --------------------------------------------------

    plt.figure(figsize=(10, 5))

    for cell in np.unique(cell_ids):

        mask = cell_ids == cell

        cell_cycles = cycle_indices[mask]
        cell_soh = y[mask]

        order = np.argsort(cell_cycles)

        plt.plot(
            cell_cycles[order],
            cell_soh[order],
            marker=".",
            markersize=3,
            linewidth=1,
            label=cell,
        )

    plt.xlabel("MATLAB cycle index")
    plt.ylabel("SOH (fraction)")
    plt.title("NASA Battery SOH vs Cycle")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    soh_path = PLOT_DIR / "soh_vs_cycle.png"

    plt.savefig(soh_path, dpi=150)
    plt.close()

    # --------------------------------------------------
    # 4. MEAN TEMPERATURE VS CYCLE
    # --------------------------------------------------

    plt.figure(figsize=(10, 5))

    for cell in np.unique(cell_ids):

        mask = cell_ids == cell

        cell_cycles = cycle_indices[mask]
        mean_temp = temperature_means[cell]

        order = np.argsort(cell_cycles)

        plt.plot(
            cell_cycles[order],
            mean_temp[order],
            marker=".",
            markersize=3,
            linewidth=1,
            label=cell,
        )

    plt.xlabel("MATLAB cycle index")
    plt.ylabel("Mean temperature (°C)")
    plt.title("NASA Mean Charge Temperature vs Cycle")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    temp_path = PLOT_DIR / "temperature_vs_cycle.png"

    plt.savefig(temp_path, dpi=150)
    plt.close()

    # --------------------------------------------------
    # 5. FINAL SUMMARY
    # --------------------------------------------------

    print("\n3. PLOTS SAVED")
    print(f"SOH plot: {soh_path}")
    print(f"Temperature plot: {temp_path}")

    print("\nDiagnostics complete.")
    print(
        "Note: Temperature statistics describe only "
        "the first 100 points of each charge sample."
    )


if __name__ == "__main__":
    main()
