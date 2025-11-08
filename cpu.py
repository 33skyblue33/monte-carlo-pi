"""
This module implements the sequential Monte Carlo algorithm for estimating PI.
"""

import argparse
import random
import sys
from tqdm import tqdm

def estimate_pi(num_samples: int) -> float:
    """
    Estimates PI using the Monte Carlo method.

    Args:
        num_samples: The number of samples to use for the estimation.

    Returns:
        The estimated value of PI.
    """
    points_inside_circle = 0
    for _ in tqdm(range(num_samples), desc="Estimating PI (Sequential CPU)"):
        x, y = random.uniform(0, 1), random.uniform(0, 1)
        if x**2 + y**2 <= 1:
            points_inside_circle += 1
    return 4 * points_inside_circle / num_samples

def main():
    """
    The main function of the module.
    """
    parser = argparse.ArgumentParser(description="Estimate PI using the Monte Carlo method (sequential).")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000000,
        help="The number of samples to use for the estimation.",
    )
    args = parser.parse_args()

    try:
        pi_estimate = estimate_pi(args.num_samples)
        print(f"Estimated value of PI: {pi_estimate}")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
