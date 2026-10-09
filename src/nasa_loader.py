
from pathlib import Path

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NASA_DIR = PROJECT_ROOT / "data" / "raw" / "nasa"

CELL_IDS = ["B0005", "B0006", "B0007", "B0018"]
N_POINTS = 100


def get_field(obj, field):
    """Read a field from a MATLAB structure."""
    if hasattr(obj, field):
        return getattr(obj, field)

    if isinstance(obj, np.void) and obj.dtype.names:
        if field in obj.dtype.names:
            return obj[field]

    if isinstance(obj, dict):
        return obj.get(field)

    raise KeyError(f"MATLAB field not found: {field}")


def as_items(value):
    """Convert MATLAB arrays into a flat Python list."""
    return list(np.asarray(value, dtype=object).reshape(-1))


def scalar(value):
    """Convert a MATLAB scalar to a Python float."""
    return float(np.asarray(value).squeeze())


def signal(data, field):
    """Return a one-dimensional numeric signal."""
    return np.asarray(get_field(data, field), dtype=float).reshape(-1)


def load_cell(mat_path, cell_id):
    """
    Load one NASA cell.

    Each input sample uses the first 100 voltage and temperature
    readings from a charge cycle. Its SOH label is based on the
    most recent preceding discharge capacity.
    """
    mat = loadmat(
        mat_path,
        squeeze_me=True,
        struct_as_record=False,
    )

    if cell_id not in mat:
        raise KeyError(f"{cell_id} not found in {mat_path.name}")

    cell = mat[cell_id]
    cycles = as_items(get_field(cell, "cycle"))

    discharge_capacities = []
    records = []

    for cycle_index, cycle in enumerate(cycles):
        cycle_type = str(get_field(cycle, "type")).strip().lower()
        data = get_field(cycle, "data")

        if cycle_type == "discharge":
            capacity = scalar(get_field(data, "Capacity"))

            if np.isfinite(capacity) and capacity > 0:
                discharge_capacities.append(
                    (cycle_index, capacity)
                )

        elif cycle_type == "charge":
            voltage = signal(data, "Voltage_measured")
            temperature = signal(data, "Temperature_measured")

            records.append({
                "cycle_index": cycle_index,
                "voltage": voltage,
                "temperature": temperature,
            })

    if not discharge_capacities:
        raise ValueError(f"No discharge capacities found for {cell_id}")

    q0 = discharge_capacities[0][1]
    samples = []
    labels = []
    metadata = []

    for record in records:
        previous = [
            item for item in discharge_capacities
            if item[0] < record["cycle_index"]
        ]

        if not previous:
            continue

        voltage = record["voltage"]
        temperature = record["temperature"]

        if len(voltage) < N_POINTS or len(temperature) < N_POINTS:
            continue

        # Shape per sample: (2, 100) = voltage, temperature
        sample = np.stack([
            voltage[:N_POINTS],
            temperature[:N_POINTS],
        ])

        discharge_index, capacity = previous[-1]
        soh = capacity / q0

        if not np.isfinite(sample).all() or not np.isfinite(soh):
            continue

        samples.append(sample)
        labels.append(soh)
        metadata.append({
            "cell_id": cell_id,
            "charge_cycle_index": record["cycle_index"],
            "discharge_cycle_index": discharge_index,
            "capacity_ah": capacity,
            "soh": soh,
        })

    if not samples:
        raise ValueError(
            f"No usable charge samples found for {cell_id}"
        )

    return (
        np.asarray(samples, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        metadata,
    )


def build_dataset():
    """Load all four cells and combine their samples."""
    all_X = []
    all_y = []
    all_metadata = []

    for cell_id in CELL_IDS:
        mat_path = NASA_DIR / f"{cell_id}.mat"

        if not mat_path.exists():
            raise FileNotFoundError(
                f"Dataset missing: {mat_path}\n"
                "Place the NASA .mat files in data/raw/nasa/."
            )

        X, y, metadata = load_cell(mat_path, cell_id)

        all_X.append(X)
        all_y.append(y)
        all_metadata.extend(metadata)

        print(
            f"{cell_id}: samples={len(y)}, "
            f"SOH range={y.min():.3f}-{y.max():.3f}"
        )

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print("\nCombined dataset")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Unique cells: {sorted(set(m['cell_id'] for m in all_metadata))}")

    return X, y, all_metadata


if __name__ == "__main__":
    X, y, metadata = build_dataset()
