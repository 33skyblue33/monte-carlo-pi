import argparse
import multiprocessing
import os
import random
import sys
from tqdm import tqdm

def simulate_points(num_points: int) -> int:
    points_inside_circle = 0
    for _ in range(num_points):
        x, y = random.uniform(0, 1), random.uniform(0, 1)
        if x**2 + y**2 <= 1:
            points_inside_circle += 1
    return points_inside_circle

def estimate_pi_parallel(num_samples: int, num_workers: int) -> float:
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