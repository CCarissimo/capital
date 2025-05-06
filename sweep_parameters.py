import numpy as np
from itertools import product
from experiment import *


def flatten(xss):
    return [x for xs in xss for x in xs]


def get_param_combos(alpha, epsilon, gamma, n_iter):
    pow2 = [2 ** i for i in range(2, 11)]
    n_processes = []

    for i in range(len(pow2)):
        i += 1
        nproc = [2 ** j for j in range(0, i)]
        n_processes.append(nproc)

    params = []
    nparams = 0

    for i, na in enumerate(pow2):
        combined = [(na, nproc) for nproc in n_processes[i]]
        params.append(combined)
        nparams += len(combined)

    params = flatten(params)

    configurations = []

    for n_agents, n_processes in params:

        if n_processes == 1:
            for elasticity in np.linspace(0.1, 0.9, 9):
                config = smallCapitalLabourExperimentConfig(
                    n_iter,
                    n_agents,
                    n_processes,
                    alpha,
                    epsilon,
                    gamma,
                    np.array([elasticity]))
                configurations.append(config)
        else:
            for rep in range(9):
                elasticity = np.random.random(size=n_processes) * 0.8 + 0.1
                config = smallCapitalLabourExperimentConfig(
                    n_iter,
                    n_agents,
                    n_processes,
                    alpha,
                    epsilon,
                    gamma,
                    elasticity)
                configurations.append(config)

    return configurations


# print(get_param_combos(0.1, 0.1, 0.1, 10))
