"""
This module implements the parallel GPU Monte Carlo algorithm for estimating PI using Numba.
"""

import argparse
import sys
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from tqdm import tqdm


@cuda.jit
def estimate_pi_kernel(rng_states, num_samples, out):
    """
    CUDA kernel for estimating PI.

    Args:
        rng_states: A device array of random number generator states.
        num_samples: The total number of samples.
        out: A device array to store the number of points inside the circle.
    """
    thread_id = cuda.grid(1)

    if thread_id < num_samples:
        x = xoroshiro128p_uniform_float32(rng_states, thread_id)
        y = xoroshiro128p_uniform_float32(rng_states, thread_id)

        if x**2 + y**2 <= 1.0:
            cuda.atomic.add(out, 0, 1)

def estimate_pi_gpu(num_samples: int, threads_per_block: int):
    """
    Estimates PI using the Monte Carlo method on the GPU.

    Args:
        num_samples: The total number of samples to use for the estimation.
        threads_per_block: The number of threads per block.

    Returns:
        The estimated value of PI.
    """
    blocks_per_grid = (num_samples + (threads_per_block - 1)) // threads_per_block
    total_threads = blocks_per_grid * threads_per_block

    # Initialize random states
    rng_states = create_xoroshiro128p_states(total_threads, seed=1)

    # Allocate output array on the device
    out = cuda.to_device(np.zeros(1, dtype=np.int32))

    # Run the kernel
    with tqdm(total=num_samples, desc="Estimating PI (GPU)") as pbar:
        estimate_pi_kernel[blocks_per_grid, threads_per_block](
            rng_states, num_samples, out
        )
        pbar.update(num_samples)

    points_inside = out.copy_to_host()[0]
    return 4 * points_inside / num_samples

def main():
    """
    The main function of the module.
    """
    parser = argparse.ArgumentParser(description="Estimate PI using the Monte Carlo method (parallel GPU).")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000000,
        help="The number of samples to use for the estimation.",
    )
    parser.add_argument(
        "--threads_per_block",
        type=int,
        default=128,
        help="The number of threads per block.",
    )
    args = parser.parse_args()

    try:
        pi_estimate = estimate_pi_gpu(args.num_samples, args.threads_per_block)
        print(f"Estimated value of PI: {pi_estimate}")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
