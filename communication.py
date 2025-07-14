# %% Modules

import os
import sys
import time

import torch
import torch.distributed as dist 

from redirect import redirect

# %% Main

if __name__ == "__main__":
    """
    Main program
    """

    # %% Create output directory for this run

    output_path = f"output"
    os.makedirs(output_path, exist_ok=True)
    redirect(path=output_path)

    # %% Print parallel run variables

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    print(f"WORLD_SIZE: {world_size}")
    print(f"RANK: {rank}")
    print(f"LOCAL_RANK: {local_rank}")
    print(f"MASTER_ADDR: {master_addr}")
    print(f"MASTER_PORT: {master_port}")

    # %% Initiate distributed run

    backend = "nccl"
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size
    )

    print(f"NCCL version : {torch.cuda.nccl.version()}")

    # %% Set seed for reproducibility

    torch.manual_seed(42)

    # %% Synchronization

    num_megabytes = 10000.0
    num_points = int(num_megabytes * (1024.0 ** 2) / 4.0)

    tensor = torch.rand(num_points, device="cuda")
    torch.cuda.synchronize()

    # %% Run benchmark

    multiplier = 1

    if rank == 0:
        print(f"{'size(MB)':>12}  {'tavg(usec)':>12}  {'tmin(usec)':>12}  {'tmax(usec)':>12}  {'avgbw(GB/sec)':>15}  {'maxbw(GB/sec)':>15}  {'minbw(GB/sec)':>15}")

    for num_megabytes in [0.10, 0.12, 0.15, 0.20, 0.32, 0.40, 0.50, 0.64, 0.80, 1.00, 1.25, 1.50, 2.00, 3.16, 4.00, 5.00, 6.40, 8.00, \
                          10.0, 12.5, 15.0, 20.0, 31.6, 40.0, 50.0, 64.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 316.0, 400.0, 500.0, 640.0, 800.0, \
                          1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3160.0, 4000.0, 5000.0, 6400.0, 8000.0]:
    
        if num_megabytes < 10.0:
            max_iter = 100 * multiplier

        elif num_megabytes < 512.0:
            max_iter = 20 * multiplier

        elif num_megabytes < 2000.0:
            max_iter = 10 * multiplier

        else:
            max_iter = 5 * multiplier
    
        num_points = int(num_megabytes * (1024.0 ** 2) / 4.0)
    
        # %% Warmup

        dist.all_reduce(tensor[0:num_points - 1], op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        dist.all_reduce(tensor[0:num_points - 1], op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
    
        t_start = time.perf_counter()
        t_1 = t_start
        t_min = float("inf")
        t_max = 0.0
    
        for idx in range(max_iter):
            dist.all_reduce(tensor[0:num_points - 1], op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            t_2 = time.perf_counter()
            t_min = min(t_min, t_2 - t_1)
            t_max = max(t_max, t_2 - t_1)
            t_1 = t_2
    
        torch.cuda.synchronize()
        t_stop = time.perf_counter()
    
        t_total = t_stop - t_start
        t_avg = t_total / max_iter
    
        avg_bw = 4.0 * 2.0e-09 * num_points * ((world_size - 1) / world_size) / t_avg
        max_bw = 4.0 * 2.0e-09 * num_points * ((world_size - 1) / world_size) / t_min
        min_bw = 4.0 * 2.0e-09 * num_points * ((world_size - 1) / world_size) / t_max
    
        if rank == 0:
            print(f"{num_megabytes:>12.2f}  {t_avg * 1.0e06:>12.2f}  {t_min * 1.0e06:>12.2f}  {t_max * 1.0e06:>12.2f}  {avg_bw:>15.2f}  {max_bw:>15.2f}  {min_bw:>15.2f}")

# %% End of script
