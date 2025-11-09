import sys
import torch 

def run_gpu_estimation(num_samples: int, threads_per_block: int, device_id: int) -> float:
    if not torch.cuda.is_available():
        print("Error: PyTorch cannot detect a CUDA-enabled GPU.", file=sys.stderr)
        sys.exit(1)

    try:
        device = torch.device(f"cuda:{device_id}")
        print(f"Using GPU device {device_id}: {torch.cuda.get_device_name(device)}")
    except Exception as e:
        print(f"Error: Could not select GPU device {device_id}. {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Number of samples: {num_samples}")

    points = torch.rand((num_samples, 2), device=device)

    dist_sq = (points ** 2).sum(dim=1)

    points_inside_circle = (dist_sq <= 1).sum()

    return 4 * points_inside_circle.item() / num_samples