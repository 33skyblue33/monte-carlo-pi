import argparse
import sys
import time
import os


def list_available_gpus():
    try:
        import torch

        if not torch.cuda.is_available():
            print(
                "PyTorch could not detect a CUDA-enabled GPU or the necessary drivers.\n"
                "Please ensure you have a supported GPU and the NVIDIA CUDA Toolkit is installed and compatible.",
                file=sys.stderr,
            )
            sys.exit(1)

        device_count = torch.cuda.device_count()
        if not device_count:
            print("CUDA is available, but no compatible GPUs were found by PyTorch.", file=sys.stderr)
            sys.exit(1)

        print("--- Available CUDA GPUs (detected by PyTorch) ---")
        for i in range(device_count):
            print(f"  Device ID {i}: {torch.cuda.get_device_name(i)}")
        print("---------------------------------------------------")
        print("Use the --device flag to select a GPU (e.g., --device 0).")

    except ImportError:
        print("PyTorch is not installed. Please install it to list GPUs.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while trying to list GPUs: {e}", file=sys.stderr)
        sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate PI using different Monte Carlo methods.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "algorithm",
        nargs="?",
        choices=["cpu", "cpu-parallel", "gpu"],
        help="The algorithm to use for the estimation.",
    )
    group.add_argument(
        "--list_gpus",
        action="store_true",
        help="List available CUDA-enabled GPUs and exit.",
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
        default=256,
        help="The number of threads per block (ignored by 'gpu' PyTorch version).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="The ID of the GPU to use for the 'gpu' algorithm.",
    )
    return parser.parse_args()


def run_algorithm(args: argparse.Namespace) -> float:
    if args.algorithm == "cpu":
        from cpu import estimate_pi

        return estimate_pi(args.num_samples)

    if args.algorithm == "cpu-parallel":
        from cpu_parallel import estimate_pi_parallel

        num_workers = args.num_workers or os.cpu_count()
        return estimate_pi_parallel(args.num_samples, num_workers)

    if args.algorithm == "gpu":
        from gpu_parallel import run_gpu_estimation

        return run_gpu_estimation(
            args.num_samples, args.threads_per_block, args.device
        )

    raise ValueError(f"Unknown algorithm: {args.algorithm}")


def main():
    args = parse_arguments()

    if args.list_gpus:
        list_available_gpus()
        sys.exit(0)

    try:
        start_time = time.time()
        pi_estimate = run_algorithm(args)
        end_time = time.time()

        print(f"Algorithm: {args.algorithm}")
        print(f"Estimated value of PI: {pi_estimate}")
        print(f"Execution time: {end_time - start_time:.4f} seconds")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()