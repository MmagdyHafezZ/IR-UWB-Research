#!/usr/bin/env python3
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def load_raw(path: str) -> np.ndarray:
    """
    Load raw data from a .npy file.
    Extend this later if you have other formats.
    """
    arr = np.load(path)
    return arr


def describe_array(name: str, arr: np.ndarray) -> None:
    print(f"=== {name} ===")
    print(f"shape      : {arr.shape}")
    print(f"dtype      : {arr.dtype}")
    print(f"is complex : {np.iscomplexobj(arr)}")
    print(f"min, max   : {arr.min()}, {arr.max()}")
    print()


def to_magnitude(arr: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(arr):
        return np.abs(arr)
    return arr.astype(np.float64)


def visualize_1d(mag: np.ndarray, out_png: str | None = None) -> None:
    """
    mag: 1D real-valued array
    """
    # Linear magnitude
    plt.figure(figsize=(10, 4))
    plt.plot(mag)
    plt.title("Raw magnitude (1D)")
    plt.xlabel("Sample index")
    plt.ylabel("Magnitude")
    plt.grid(True)
    if out_png is not None:
        plt.tight_layout()
        plt.savefig(out_png.replace(".png", "_1d_lin.png"), dpi=300)
    plt.show()

    # Log magnitude (dB)
    mag_log = 20 * np.log10(mag + 1e-12)
    plt.figure(figsize=(10, 4))
    plt.plot(mag_log)
    plt.title("Raw magnitude (1D, dB)")
    plt.xlabel("Sample index")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True)
    if out_png is not None:
        plt.tight_layout()
        plt.savefig(out_png.replace(".png", "_1d_db.png"), dpi=300)
    plt.show()


def visualize_2d(mag: np.ndarray, out_png: str | None = None) -> None:
    """
    mag: 2D real-valued array. We treat axis 0 as "slow time" and axis 1 as "fast time".
    If your interpretation is opposite, just transpose before calling.
    """
    mag_log = 20 * np.log10(mag + 1e-12)

    plt.figure(figsize=(8, 8))
    plt.imshow(
        mag_log,
        aspect="auto",
        cmap="inferno",
        origin="lower",
    )
    plt.title("Range–time image (log magnitude)")
    plt.xlabel("Fast-time samples (range bins)")
    plt.ylabel("Slow-time index (pulse no.)")
    plt.colorbar(label="Magnitude [dB]")
    plt.tight_layout()
    if out_png is not None:
        plt.savefig(out_png, dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Read and visualize IR-UWB raw data (.npy).")
    parser.add_argument("path", help="Path to raw_data.npy (complex or real)")
    args = parser.parse_args()

    path = args.path
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    arr = load_raw(path)
    describe_array("Loaded array", arr)

    mag = to_magnitude(arr)
    describe_array("Magnitude", mag)

    base_png = os.path.splitext(path)[0] + "_viz.png"

    if mag.ndim == 1:
        visualize_1d(mag, out_png=base_png)
    elif mag.ndim == 2:
        # If it looks "tall and skinny", flip axes so pulses are vertical index
        if mag.shape[0] < mag.shape[1]:
            # shape (num_samples, num_pulses) -> transpose
            mag_for_plot = mag.T
        else:
            mag_for_plot = mag
        visualize_2d(mag_for_plot, out_png=base_png)
    else:
        print(f"Array has {mag.ndim} dimensions – not handled explicitly.")
        print("You can reshape or squeeze dimensions before plotting.")


if __name__ == "__main__":
    main()
