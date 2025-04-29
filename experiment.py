import numpy as np
from processes import *
from agents import *
from dataclasses import dataclass
from typing import Union
import copy


@dataclass
class capitalLabourExperimentConfig:
    n_iter: int
    n_agents: int
    n_processes: int
    alpha: float
    epsilon: float
    gamma: float
    wants: Union[int, np.ndarray]
    capitals: Union[int, np.ndarray]
    timenergy: Union[int, np.ndarray]
    p_multipliers: np.ndarray
    p_elasticities: np.ndarray


def run_capital_labour_processes(n_iter, epsilon, alpha, gamma, n_agents, n_processes, wants, capitals, timenergy, p_multipliers, p_elasticities):

    Q_capital = np.random.random(size=(n_agents, 1, n_processes)) * 0.1
    Q_labour = np.random.random(size=(n_agents, 1, n_processes)) * 0.1
    S = np.zeros((n_agents)).astype(int)
    redistribution_thresholds = p_multipliers

    M = {}

    for t in range(n_iter):

        # print("iteration", t)

        roles = np.where(wants > capitals, 0, 1)
        actions = np.where(roles == 1, e_greedy_select_action(Q_capital, S, epsilon), 0)

        capitalists = np.nonzero(roles == 1)[0]
        labourers = np.nonzero(roles == 0)[0]

        # print("capitalists", capitalists)

        capital_kinetic = capitals[capitalists]
        labour_kinetic = timenergy[labourers]

        # print("kinetic capital", capital_kinetic)

        capitalists_actions = actions[capitalists]
        # labourers_actions = actions[labourers]

        # print("capital actions", capitalists_actions)
        # print("labour actions", labourers_actions)

        capital_allocations = np.array(
            [np.where(capitalists_actions == i, capital_kinetic, 0).sum() for i in range(n_processes)])

        funded_processes = np.nonzero(capital_allocations > 0)[0]
        # print(funded_processes)

        if len(funded_processes) == 0:
            rewards = np.zeros(n_agents)
            produced = np.zeros(n_processes)
            labour_allocations = np.zeros(n_processes)

        else:  # at least one process funded
            # print(funded_processes)

            actions = np.where(roles == 0, e_greedy_select_action(Q_labour, S, epsilon, funded_processes), actions)
            labourers_actions = actions[labourers]

            labour_allocations = np.array(
                [np.where(labourers_actions == i, labour_kinetic, 0).sum() for i in range(n_processes)])

            worked_processes = np.nonzero(labour_allocations > 0)[0]

            #  print("capitals", capitals)

            mask = np.isin(capitalists_actions, worked_processes)
            capitals[capitalists] = np.where(mask, 0, capitals[capitalists])

            # print("capitals", capitals)

            # print("capital to process", capital_allocations)
            #  print("labour to process", labour_allocations)

            produced = [production(m, e, c, l) for m, e, c, l in
                        list(zip(p_multipliers, p_elasticities, capital_allocations, labour_allocations))]
            #  print("produced", produced)

            returns = np.array([redistribution(Y, e, c, l) for Y, e, c, l in
                                list(zip(produced, p_elasticities, capital_allocations, labour_allocations))])
            #  print("returns", returns)

            rewards = np.zeros(n_agents)
            rewards[labourers] = np.array([returns[a, 0] for a in actions[labourers]])
            rewards[capitalists] = np.array([returns[a, 1] for a in actions[capitalists]])

        # print("rewards", rewards)

        # print(capitalists, Q_capital, S, actions, rewards, alpha, gamma)

        Q_capital, _ = bellman_update_q_table(capitalists, Q_capital, S, actions, rewards, S, alpha, gamma)
        Q_labour, _ = bellman_update_q_table(labourers, Q_labour, S, actions, rewards, S, alpha, gamma)

        capitals[capitalists] += rewards[capitalists] * capital_kinetic
        capitals[labourers] += rewards[labourers] * labour_kinetic

        capitals = np.max(np.vstack([capitals - wants, np.zeros(n_agents)]), axis=0)

        # print("final capitals", capitals)
        n_capitalists = np.sum(roles)
        n_labourers = n_agents - n_capitalists

        M[t] = {
            "A": actions,
            "Y": produced,
            "C": capitals,
            "Cp": capital_allocations,
            "Lp": labour_allocations,
            "R": rewards,
            "nL": n_labourers,
            "nC": n_capitalists,
            "E": copy.deepcopy(p_elasticities),
            "W": wants
        }

        ## evolution steps

        wants = evolve_agent_wants(capitals, wants)
        p_elasticities, dead_process_indices = evolve_processes(p_elasticities, capital_allocations)
        Q_capital = q_table_replace_process(Q_capital, dead_process_indices)
        Q_labour = q_table_replace_process(Q_labour, dead_process_indices)

        print(p_elasticities)

    return M


if __name__ == "__main__":

    from plotting import plot_dashboard
    # from analysis import run_analysis

    n_agents = 100
    n_processes = 10
    
    # wants = np.array([10.0, 100.0])
    # capitals = np.array([100.0, 10.0])
    # timenergy = np.array([50, 50])
    # p_multipliers = np.array([4])
    # p_elasticities = np.array([0.8])
    
    wants = np.random.randint(1, 100, size=n_agents).astype(float)
    capitals = np.random.randint(1, 100, size=n_agents).astype(float)
    timenergy = np.ones(n_agents)*50
    p_multipliers = np.random.random(size=n_processes)*10
    p_elasticities = np.random.random(size=n_processes)  # np.array([0.04765849, 0.04537723])

    
    n_iter = 1000
    epsilon = 0.1
    alpha = 0.1
    gamma = 0

    M = run_capital_labour_processes(
        n_iter,
        epsilon,
        alpha,
        gamma,
        n_agents,
        n_processes,
        wants,
        capitals,
        timenergy,
        p_multipliers,
        p_elasticities
    )

    plot_dashboard(M, save=True)

    # run_analysis(M, capitals, wants)
