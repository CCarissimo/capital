import argparse
import os
from parallel_simulation import multi_file_simulation, flatten
from experiment import capitalLabourExperimentConfig
import itertools
import pandas as pd
import numpy as np
from sweep_parameters import get_param_combos


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="input simulation parameters")

    # Add arguments
    parser.add_argument('alpha', type=float, help="learning rate")
    parser.add_argument('epsilon', type=float, help="exploration rate")
    parser.add_argument('gamma', type=float, help="discount factor")
    # parser.add_argument('directory', type=str, help="location where to save files")

    # Parse the arguments
    args = parser.parse_args()

    main_dir = "/cluster/work/coss/ccarissimo/capital_labour_processes/"
    # main_dir = "test_multiprocessing/"
    data_addr = f"{main_dir}data/"
    if not os.path.isdir(data_addr):
        os.mkdir(data_addr)
    dataframes_addr = f"{main_dir}dataframes/"
    if not os.path.isdir(dataframes_addr):
        os.mkdir(dataframes_addr)

    num_cpus = int(os.environ.get("SLURM_NTASKS", os.cpu_count()))  # specific for euler cluster
    print("identified cpus", num_cpus)

    n_iter = [10 ** 1]  # I suggest to reduce it to 10**4

    settings = get_param_combos(args.alpha, args.epsilon, args.gamma, n_iter)

    # print(settings)
    repeat_count = 10
    results = multi_file_simulation(settings, data_addr, repeat_count, num_processes=num_cpus)

    results = flatten(results)
    df = pd.DataFrame(results)
    filename = f"player1params_a({args.alpha})_e({args.epsilon})_g({args.gamma}).csv"
    destination = dataframes_addr + filename
    df.to_csv(destination)
