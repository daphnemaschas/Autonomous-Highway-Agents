import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window > len(values):
        window = len(values)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning curve from a .npy rewards file")
    parser.add_argument("npy_path", type=Path, help="Path to rewards .npy file")
    parser.add_argument("--window", type=int, default=50, help="Smoothing window (moving average)")
    parser.add_argument("--title", type=str, default="Learning Curve", help="Plot title")
    parser.add_argument("--save", type=Path, default=None, help="Optional output image path")
    args = parser.parse_args()

    rewards = np.load(args.npy_path)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)

    plt.figure(figsize=(10, 5))
    episodes = np.arange(1, len(rewards) + 1)
    plt.plot(episodes, rewards, alpha=0.35, linewidth=1.0, label="Episode reward")

    smooth = moving_average(rewards, args.window)
    smooth_x = np.arange(len(rewards) - len(smooth) + 1, len(rewards) + 1)
    plt.plot(smooth_x, smooth, linewidth=2.0, label=f"Moving average (window={args.window})")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(args.title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.save is not None:
        plt.savefig(args.save, dpi=150)
        print(f"Saved plot to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()