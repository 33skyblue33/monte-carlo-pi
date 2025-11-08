# Monte Carlo PI Estimation

This project provides three different implementations of the Monte Carlo method for estimating the value of PI:

1.  **Sequential CPU:** A single-threaded implementation that runs on the CPU.
2.  **Parallel CPU:** A multi-threaded implementation that leverages multiple CPU cores to speed up the estimation.
3.  **Parallel GPU:** A massively parallel implementation that runs on the GPU using the Numba library for JIT compilation.

## How it Works

The Monte Carlo method for estimating PI is a probabilistic algorithm that relies on the principles of the Monte Carlo simulation. The basic idea is to inscribe a circle within a square and then randomly generate a large number of points within the square. The ratio of the number of points that fall inside the circle to the total number of points generated is approximately equal to the ratio of the area of the circle to the area of the square.

Since the area of the circle is πr² and the area of the square is (2r)² = 4r², the ratio of the areas is πr² / 4r² = π/4. Therefore, we can estimate PI as:

PI ≈ 4 * (number of points inside the circle) / (total number of points)

## Setup

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    ```
2.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Algorithms

The `main.py` script is the entry point for running all three algorithms. You can select the algorithm to run using the first positional argument.

### Sequential CPU

To run the sequential CPU implementation, use the `cpu` algorithm:

```bash
python3 main.py cpu --num_samples <number_of_samples>
```

### Parallel CPU

To run the parallel CPU implementation, use the `cpu-parallel` algorithm. You can optionally specify the number of workers to use with the `--num_workers` flag. If not specified, the number of available CPU cores will be used.

```bash
python3 main.py cpu-parallel --num_samples <number_of_samples> --num_workers <number_of_workers>
```

### Parallel GPU

To run the parallel GPU implementation, use the `gpu` algorithm. You can optionally specify the number of threads per block to use with the `--threads_per_block` flag.

```bash
python3 main.py gpu --num_samples <number_of_samples> --threads_per_block <threads_per_block>
```

**Note:** If you do not have a CUDA-enabled GPU, you can still run the GPU implementation using the Numba CUDA simulator. To do this, set the `NUMBA_ENABLE_CUDASIM` environment variable to `1`:

```bash
NUMBA_ENABLE_CUDASIM=1 python3 main.py gpu --num_samples <number_of_samples>
```
