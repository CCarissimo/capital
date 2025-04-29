import numpy as np


def bellman_update_q_table(ind, Q, S, A, R, S_, alpha, gamma):
    """
    Performs a one-step update using the bellman update equation for Q-learning.
    :param ind: the indices of agents to update for, must match shapes of other arrays
    :param agentConfig:
    :param Q: np.ndarray Q-table indexed by (agents, states, actions)
    :param S: np.ndarray States indexed by (agents)
    :param A: np.ndarray Actions indexed by (agents)
    :param R: np.ndarray Rewards indexed by (agents)
    :param S_: np.ndarray Next States indexed by (agents)
    :return: np.ndarray Q-table indexed by (agents, states, actions)
    """
    # print(ind, Q[ind, S[ind], A[ind]], S[ind], A[ind], R[ind], S_[ind], Q[ind, S_[ind]].shape)
    all_belief_updates = alpha * (R[ind] + gamma * Q[ind, S_[ind]].max(axis=1) - Q[ind, S[ind], A[ind]])
    Q[ind, S[ind], A[ind]] = Q[ind, S[ind], A[ind]] + all_belief_updates
    return Q, np.abs(all_belief_updates).sum()


def e_greedy_select_action(Q, S, epsilon, indices=None):
    """
    Select actions based on an epsilon-greedy policy.
    Q: (agents, states, actions)
    S: (agents,)
    epsilon: float
    indices: np.ndarray or None, indices of allowed actions
    """
    n_agents = Q.shape[0]

    if indices is None:
        indices = np.arange(Q.shape[-1])  # all actions allowed

    rand = np.random.random_sample(size=n_agents)
    randA = np.random.choice(indices, size=n_agents, replace=True)

    # Select Q-values restricted to allowed indices
    q_values = Q[np.arange(n_agents), S, :][:, indices]  # shape (n_agents, len(indices))

    greedyA_idx = np.argmax(q_values, axis=1)  # index relative to `indices`
    greedyA = indices[greedyA_idx]  # map back to original action indices

    # Epsilon-greedy selection
    A = np.where(rand >= epsilon, greedyA, randA)

    return A
