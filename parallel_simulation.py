#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
import os
from tqdm import tqdm
from dataclasses import asdict
from experiment import *
from processes import gini, theoretical_max_production


def params_dirname(alpha, epsilon, gamma):
    return f"/params_a({alpha})_e({epsilon})_g({gamma})/"


def run_and_store_one_setting(args):

    base_addr, repeat_count, params = args

    # extract player 1 parameters to create directory for player 1 settings
    sub_directory_name = params_dirname(params.alpha, params.epsilon, params.gamma)
    save_path = base_addr + sub_directory_name
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

    extracted_records = []
    for i in range(repeat_count):
        run_results = run_small_capital_labour_processes(
            params.n_iter,
            params.n_agents,
            params.n_processes,
            params.alpha,
            params.epsilon,
            params.gamma,
            params.p_elasticities,
        )

        extracted_records.append(record2df(run_results, params, i))

    return extracted_records


def record2df(record, params, repeat_no):
    frame = asdict(params)

    exclusion_threshold = 0.5
    times = sorted(record.keys())[int(len(record.keys())*exclusion_threshold):-1]

    Y = np.array([record[t]["Y"] for t in times])
    Yall = np.mean(Y)
    Ymean = np.mean(Y, axis=0)
    Ymedian = np.median(Y, axis=0)
    Ystd = np.std(Y, axis=0)
    Ymax = np.max(Y, axis=0)

    rewards = np.array([record[t]["R"] for t in times])
    roles = np.array([record[t]["roles"] for t in times])
    avg_reward_capital = (rewards * roles).mean()
    avg_reward_labour = (rewards * (1-roles)).mean()
    max_reward_capital = (rewards * roles).max()

    max_labour = params.n_agents * 100  # fixed at 100 timenergy
    Yoptimum = theoretical_max_production(2, params.p_elasticities, max_labour)  # multipliers fixed at 2

    C = np.array([np.mean(record[t]["C"]) for t in times])
    Cmean = np.mean(C)
    Cmedian = np.median(C)
    Cstd = np.std(C)

    nL = np.array([np.mean(record[t]["nL"]) for t in times])
    nLmean = np.mean(nL)
    nLmedian = np.median(nL)
    nLstd = np.std(nL)

    nC = np.array([np.mean(record[t]["nC"]) for t in times])
    nCmean = np.mean(nC)
    nCmedian = np.median(nC)
    nCstd = np.std(nC)

    G = np.array([gini(record[t]["C"]) for t in times])
    Gmean = np.mean(G)
    Gmedian = np.median(G)
    Gstd = np.std(G)

    row = {
        "repetition": repeat_no,
        "Yopt": Yoptimum,
        "Y": Ymean,
        "Yall": Yall,
        "Ymedian": Ymedian,
        "Ystd": Ystd,
        "Ymax": Ymax,
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
        "avgRcapital": avg_reward_capital,
        "avgRlabour": avg_reward_labour,
        "maxRcapital": max_reward_capital,
        "lastt": times[-1]
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
    n_agents = [10]
    n_processes = [1]
    alpha = [0.1]
    epsilon = [0.01]
    gamma = [0.1]
    elasticities = [np.array([0.5])]

    settings = [
        smallCapitalLabourExperimentConfig(I, N, P, a, e, g, pE)
        for I, N, P, a, e, g, pE in itertools.product(
            n_iter,
            n_agents,
            n_processes,
            alpha,
            epsilon,
            gamma,
            elasticities
        )
    ]

    num_cpus = int(os.cpu_count())  # specific for euler cluster
    print("identified cpus", num_cpus)
    worker_count = num_cpus

    addr = "/Users/cesarecarissimo/data/test_capital/"

    repeat_count = 40
    results = multi_file_simulation(settings, addr, repeat_count, num_processes=num_cpus)

    results = flatten(results)
    df = pd.DataFrame(results)
    destination = "/Users/cesarecarissimo/data/test_capital/test_outfile.csv"
    df.to_csv(destination)

    # convert_directory(addr, "/cluster/home/ccarissimo/Bachelors_Project_Simulations/Utils/Simulations/duopoly_5_full_sweep_10e6_2.csv", RecordUtils.RecordMode.PICKLE)
