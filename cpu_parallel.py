"""
This module implements the parallel CPU Monte Carlo algorithm for estimating PI.
"""

import argparse
import multiprocessing
import os
import random
import sys
from tqdm import tqdm

def simulate_points(num_points: int) -> int:
    """
    Simulates a given number of points and returns the count of points inside the unit circle.

    Args:
        num_points: The number of points to simulate.

    Returns:
        The number of points that fall inside the unit circle.
    """
    points_inside_circle = 0
    for _ in range(num_points):
        x, y = random.uniform(0, 1), random.uniform(0, 1)
        if x**2 + y**2 <= 1:
            points_inside_circle += 1
    return points_inside_circle

def estimate_pi_parallel(num_samples: int, num_workers: int) -> float:
    """
    Estimates PI using the Monte Carlo method in parallel on the CPU.

    Args:
        num_samples: The total number of samples to use for the estimation.
        num_workers: The number of parallel processes to use.

    Returns:
        The estimated value of PI.
    """
    pool = multiprocessing.Pool(processes=num_workers)
    try:
        samples_per_worker = [num_samples // num_workers] * num_workers
        remainder = num_samples % num_workers
        for i in range(remainder):
            samples_per_worker[i] += 1
        total_points_inside_circle = 0

        with tqdm(total=num_samples, desc=f"Estimating PI (Parallel CPU with {num_workers} workers)") as pbar:
            for i, points_in_chunk in enumerate(pool.imap_unordered(simulate_points, samples_per_worker)):
                total_points_inside_circle += points_in_chunk
                pbar.update(samples_per_worker[i])

        return 4 * total_points_inside_circle / num_samples
    finally:
        pool.close()
        pool.join()

def main():
    """
    The main function of the module.
    """
    parser = argparse.ArgumentParser(description="Estimate PI using the Monte Carlo method (parallel CPU).")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000000,
        help="The number of samples to use for the estimation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="The number of parallel processes to use.",
    )
    args = parser.parse_args()

    try:
        pi_estimate = estimate_pi_parallel(args.num_samples, args.num_workers)
        print(f"Estimated value of PI: {pi_estimate}")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
