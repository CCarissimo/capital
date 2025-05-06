#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
import re
import os
from tqdm import tqdm
import pickle
from dataclasses import asdict
from experiment import *
# from learning_in_games.games import braess_augmented_network
from processes import gini


def player1params_dirname(PLACEHOLDER):
    return f"/params_a({PLACEHOLDER})/"


def run_and_store_one_setting(args):

    base_addr, repeat_count, params = args
    records = {}
    file_name = str(params)

    # extract player 1 parameters to create directory for player 1 settings
    sub_directory_name = player1params_dirname(params.alpha)  #TODO
    save_path = base_addr + sub_directory_name
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

    # create tmp directory
    # os.makedirs(save_path + "/" + file_name)
    extracted_records = []
    for i in range(repeat_count):
        run_results = run_capital_labour_processes(
            params.n_iter,
            params.n_agents,
            params.n_processes,
            params.alpha,
            params.epsilon,
            params.gamma,
            params.wants,
            params.capitals,
            params.timenergy,
            params.p_multipliers,
            params.p_elasticities,
        )
        # records[i] = run_results
        extracted_records.append(record2df(run_results, params, i))

        # q_table = run_results[params.n_iter]["Q"]
        # np.savez_compressed(f"{save_path}{file_name}_run{i}", a=q_table)

    # records["params"] = params

    # with open(f"{save_path}{file_name}.pkl", "wb") as file:
    #     pickle.dump(records, file)

    return extracted_records


def record2df(record, params, repeat_no):
    frame = asdict(params)

    exclusion_threshold = 0.2
    times = sorted(record.keys())[int(len(record.keys())*0.2):-1]

    Y = np.array([record[t]["Y"].mean() for t in times])
    Ymean = np.mean(Y)
    Ymedian = np.median(Y)
    Ystd = np.std(Y)

    C = np.array([record[t]["C"].mean() for t in times])
    Cmean = np.mean(C)
    Cmedian = np.median(C)
    Cstd = np.std(C)

    nL = np.array([record[t]["nL"].mean() for t in times])
    nLmean = np.mean(nL)
    nLmedian = np.median(nL)
    nLstd = np.std(nL)

    nC = np.array([record[t]["nC"].mean() for t in times])
    nCmean = np.mean(nC)
    nCmedian = np.median(nC)
    nCstd = np.std(nC)

    G = np.array([gini(record[t]["C"]) for t in times])
    Gmean = np.mean(G)
    Gmedian = np.median(G)
    Gstd = np.std(G)

    row = {
        "repetition": repeat_no,
        "Y": Ymean,
        "Ymedian": Ymedian,
        "Ystd": Ystd,
        "C": Cmean,
        "Cmedian": Cmedian,
        "Cstd": Cstd,
        "nL": nLmean,
        "nLmedian": nLmedian,
        "nLstd": nLstd,
        "nC": nCmean,
        "nCmedian": nCmedian,
        "nCstd": nCstd,
        "G": Gmean,
        "Gmedian": Gmedian,
        "Gstd": Gstd,
    }

    frame.update(row)

    return frame


def multi_file_simulation(settings, base_addr, repeat_count, num_processes):
    args_list = [[base_addr, repeat_count, params] for params in settings]
    return run_apply_async_multiprocessing(run_and_store_one_setting, args_list, num_processes)


def run_apply_async_multiprocessing(func, argument_list, num_processes):
    pool = mp.Pool(processes=num_processes)

    jobs = [
        pool.apply_async(func=func, args=(*argument,)) if isinstance(argument, tuple) else pool.apply_async(func=func,
                                                                                                            args=(
                                                                                                                argument,))
        for argument in argument_list]
    pool.close()
    result_list_tqdm = []
    for job in tqdm(jobs):
        result_list_tqdm.append(job.get())

    return result_list_tqdm


def flatten(xss):
    return [x for xs in xss for x in xs]


if __name__ == '__main__':
    n_iter = [10 ** 2]  # I suggest to reduce it to 10**4
    n_agents = [100]
    n_processes = [10]
    alpha = [0.1]
    epsilon = [0.01]
    gamma = [0.1]
    wants = []
    capitals = []
    timenergy = []
    multipliers = []
    elasticities = []

    settings = [
        capitalLabourExperimentConfig(I, N, P, a, e, g, W, C, T, pM, pE)
        for I, N, P, a, e, g, W, C, T, pM, pE in itertools.product(
            n_iter,
            n_agents,
            n_processes,
            alpha,
            epsilon,
            gamma,
            wants,
            capitals,
            timenergy,
            multipliers,
            elasticities
        )
    ]

    num_cpus = int(os.cpu_count())  # specific for euler cluster
    print("identified cpus", num_cpus)
    worker_count = num_cpus

    addr = "/Users/ccarissimo/data/test_braess/data"

    repeat_count = 40
    results = multi_file_simulation(settings, addr, repeat_count, num_processes=num_cpus)

    results = flatten(results)
    df = pd.DataFrame(results)
    destination = "/Users/ccarissimo/data/test_braess/test_outfile.csv"
    df.to_csv(destination)

    # convert_directory(addr, "/cluster/home/ccarissimo/Bachelors_Project_Simulations/Utils/Simulations/duopoly_5_full_sweep_10e6_2.csv", RecordUtils.RecordMode.PICKLE)
