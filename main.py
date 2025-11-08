"""
This is the main module for running the PI estimation algorithms.
"""

import argparse
import sys
import time
from cpu import estimate_pi as estimate_pi_cpu
from cpu_parallel import estimate_pi_parallel as estimate_pi_cpu_parallel
from gpu_parallel import estimate_pi_gpu

def main():
    """
    The main function of the module.
    """
    parser = argparse.ArgumentParser(description="Estimate PI using different Monte Carlo methods.")
    parser.add_argument(
        "algorithm",
        choices=["cpu", "cpu-parallel", "gpu"],
        help="The algorithm to use for the estimation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000000,
        help="The number of samples to use for the estimation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="The number of parallel processes to use (for cpu-parallel).",
    )
    parser.add_argument(
        "--threads_per_block",
        type=int,
        default=128,
        help="The number of threads per block (for gpu).",
    )
    args = parser.parse_args()

    start_time = time.time()
    pi_estimate = 0

    try:
        if args.algorithm == "cpu":
            pi_estimate = estimate_pi_cpu(args.num_samples)
        elif args.algorithm == "cpu-parallel":
            import os
            num_workers = args.num_workers if args.num_workers is not None else os.cpu_count()
            pi_estimate = estimate_pi_cpu_parallel(args.num_samples, num_workers)
        elif args.algorithm == "gpu":
            pi_estimate = estimate_pi_gpu(args.num_samples, args.threads_per_block)

        end_time = time.time()

        print(f"Algorithm: {args.algorithm}")
        print(f"Estimated value of PI: {pi_estimate}")
        print(f"Execution time: {end_time - start_time:.4f} seconds")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
