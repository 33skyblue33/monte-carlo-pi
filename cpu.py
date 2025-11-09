import argparse
import random
import sys
from tqdm import tqdm

def estimate_pi(num_samples: int) -> float:
    points_inside_circle = 0
    for _ in tqdm(range(num_samples), desc="Estimating PI (Sequential CPU)"):
        x, y = random.uniform(0, 1), random.uniform(0, 1)
        if x**2 + y**2 <= 1:
            points_inside_circle += 1
    return 4 * points_inside_circle / num_samples