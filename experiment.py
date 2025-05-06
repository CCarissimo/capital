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


def run_capital_labour_processes(n_iter, epsilon, alpha, gamma, n_agents, n_processes, wants, capitals, timenergy,
                                 p_multipliers, p_elasticities, p_redistribution):

    Q_capital = np.random.random(size=(n_agents, 1, n_processes)) * 100
    Q_labour = np.random.random(size=(n_agents, 1, n_processes)) * 100
    S = np.zeros((n_agents)).astype(int)
    redistribution_thresholds = p_elasticities

    M = {}
    death_counter = 0

    for t in range(n_iter):

        if np.sum(capitals) == 0:
            death_counter += 1
            if death_counter > 100:
                # print(np.sum(roles))
                break

        surplus = capitals - wants

        potential_capitalists = np.where(surplus > 0, 1, 0)
        Q_combined = np.stack([Q_labour.max(axis=2), Q_capital.max(axis=2)], axis=-1)
        # argmax_roles = Q_combined.argmax(axis=-1).flatten()
        argmax_roles = e_greedy_select_action(Q_combined, S, epsilon)

        # labourers are 0 and capitalists are 1
        roles = np.where(potential_capitalists == 1, argmax_roles, 0)
        capitalists = np.nonzero(roles == 1)[0]
        labourers = np.nonzero(roles == 0)[0]

        capital_kinetic = surplus[capitalists]
        labour_kinetic = timenergy[labourers]

        actions = np.where(roles == 1, e_greedy_select_action(Q_capital, S, epsilon), 0)

        capitalists_actions = actions[capitalists]

        capital_allocations = np.array(
            [np.where(capitalists_actions == i, capital_kinetic, 0).sum() for i in range(n_processes)])

        funded_processes = np.nonzero(capital_allocations > 0)[0]

        if len(funded_processes) == 0:
            rewards = np.zeros(n_agents)
            produced = np.zeros(n_processes)
            labour_allocations = np.zeros(n_processes)

        else:  # at least one process funded
            actions = np.where(roles == 0, e_greedy_select_action(Q_labour, S, epsilon, indices=None), actions)
            labourers_actions = actions[labourers]

            labour_allocations = np.array(
                [np.where(labourers_actions == i, labour_kinetic, 0).sum() for i in range(n_processes)])

            worked_processes = np.nonzero(labour_allocations > 0)[0]

            if len(worked_processes) == 0:
                rewards = np.zeros(n_agents)
                produced = np.zeros(n_processes)

            else:  # at least one process was worked
                mask = np.isin(capitalists_actions, worked_processes)
                capitals[capitalists] = np.where(mask, capitals[capitalists] - capital_kinetic, capitals[capitalists])

                print(p_multipliers, p_elasticities, capital_allocations, labour_allocations)

                produced = [production(m, e, c, l) for m, e, c, l in
                            list(zip(p_multipliers, p_elasticities, capital_allocations, labour_allocations))]

                returns = np.array([redistribution(Y, e, c, l, np.sum(1 - roles)) for Y, e, c, l in
                                    list(zip(produced, redistribution_thresholds, capital_allocations,
                                             labour_allocations))])

                rewards = np.zeros(n_agents)
                rewards[labourers] = np.array([returns[a, 0] for a in actions[labourers]])
                rewards[capitalists] = np.array([returns[a, 1] for a in actions[capitalists]])

                allocations = np.zeros(n_agents)
                allocations[labourers] = labour_kinetic
                allocations[capitalists] = capital_kinetic
                rewards[capitalists] *= allocations[capitalists]
                # rewards[labourers] /= np.sum(1 - roles)

        Q_capital, _ = bellman_update_q_table(capitalists, Q_capital, S, actions, rewards, S, alpha, gamma)
        Q_labour, _ = bellman_update_q_table(labourers, Q_labour, S, actions, rewards, S, alpha, gamma)

        capitals[capitalists] += rewards[capitalists]  # * capital_kinetic
        capitals[labourers] += rewards[labourers]  # * labour_kinetic

        capitals = np.max(np.vstack([capitals - wants, np.zeros(n_agents)]), axis=0)

        n_capitalists = np.sum(roles)
        n_labourers = n_agents - n_capitalists

        M[t] = {
            "A": actions,
            "Y": produced,
            "C": copy.deepcopy(capitals),
            "Cp": capital_allocations,
            "Lp": labour_allocations,
            "R": copy.deepcopy(rewards),
            "nL": n_labourers,
            "nC": n_capitalists,
            "E": copy.deepcopy(p_elasticities),
            "W": wants,
            "roles": copy.deepcopy(roles)
        }

    return M


@dataclass
class smallCapitalLabourExperimentConfig:
    n_iter: int
    n_agents: int
    n_processes: int
    alpha: float
    epsilon: float
    gamma: float
    p_elasticities: np.ndarray


def run_small_capital_labour_processes(n_iter, n_agents, n_processes, alpha, epsilon, gamma, p_elasticities):

    capitals = np.ones(n_agents)*100
    timenergy = np.ones(n_agents)*100
    p_multipliers = np.ones(n_processes)*2
    wants = np.zeros(n_agents)

    M = run_capital_labour_processes(n_iter, epsilon, alpha, gamma, n_agents, n_processes, wants, capitals, timenergy,
                                 p_multipliers, p_elasticities, p_multipliers)

    return M


if __name__ == "__main__":

    from plotting import plot_dashboard
    # from analysis import run_analysis
    
    # n_agents = 2
    # n_processes = 1
    # wants = np.array([10.0, 100.0])
    # capitals = np.array([100.0, 10.0])
    # timenergy = np.array([50, 50])
    # p_multipliers = np.array([4])
    # p_elasticities = np.array([0.8])

    n_agents = 4
    n_processes = 1
    wants = np.zeros(n_agents)  # np.random.randint(0, 100, size=n_agents).astype(float)
    capitals = np.ones(n_agents) * 100  # np.random.randint(1, 50, size=n_agents).astype(float)
    timenergy = np.ones(n_agents) * 100
    p_multipliers = np.ones(n_processes) * 2  # np.random.random(size=n_processes)
    
    p_elasticities = np.ones(n_processes) * 0.25
    # p_elasticities = np.clip(np.random.random(size=n_processes), 0.1, 0.9)
    
    p_redistribution = p_elasticities

    n_iter = 10000
    epsilon = 0.1
    alpha = 0.1
    gamma = 0

    print("Max Production")
    print(theoretical_max_production(p_multipliers, p_elasticities, timenergy.sum()))

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
        p_elasticities,
        p_redistribution
    )

    plot_dashboard(M, save=True)

    # run_analysis(M, capitals, wants)
