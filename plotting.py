import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from processes import gini


def plot_dashboard(M, save=False):
    times = sorted(M.keys())  # sort timestamps
    n_times = len(times)

    fig, axs = plt.subplots(2, 5, figsize=(12, 7))
    axs = axs.flatten()

    # Collect time series
    actions = np.array([M[t]["A"] for t in times])  # (n_times, n_agents)
    produced = [M[t]["Y"] for t in times]  # list of arrays (size n_processes)
    capitals = np.array([M[t]["C"] for t in times])  # (n_times, n_agents)
    rewards = np.array([M[t]["R"] for t in times])  # (n_times, n_agents)
    elasticities = np.array([M[t]["E"] for t in times])
    wants = np.array([M[t]["W"] for t in times])

    n_labourers = np.array([M[t]["nL"] for t in times])
    n_capitalists = np.array([M[t]["nC"] for t in times])
    
    # --- Plot actions ---
    sns.heatmap(actions.T, ax=axs[0], cbar=True)
    axs[0].set_title("Actions over time")
    axs[0].set_ylabel("Agent")
    axs[0].set_xlabel("Time")

    # --- Plot produced ---
    max_processes = max(len(y) for y in produced)
    produced_padded = np.full((n_times, max_processes), np.nan)
    for i, y in enumerate(produced):
        produced_padded[i, :len(y)] = y
    axs[1].plot(times, produced_padded)
    axs[1].set_title("Produced per process")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Produced")
    axs[1].set_yscale("log")
    # axs[1].set_ylim((10e-2, max(produced_padded)))

    # --- Plot capitals ---
    if capitals.shape[1] > 10:
        avg_capitals = np.mean(capitals, axis=1)
        std_capitals = np.std(capitals, axis=1)
        axs[2].plot(times, avg_capitals)
        axs[2].fill_between(times, avg_capitals - std_capitals, avg_capitals + std_capitals, color='blue', alpha=0.2, label='±1 Std Dev')
    else:
        axs[2].plot(times, capitals)
    
    axs[2].set_title("Capitals per agent")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Capital")
    axs[2].set_yscale("log")

    # --- Plot capital allocations ---
    cap_alloc_sizes = [M[t]["Cp"] for t in times]
    axs[3].plot(times, cap_alloc_sizes, marker='o')
    axs[3].set_title("Capital allocations (number)")
    axs[3].set_xlabel("Time")
    axs[3].set_ylabel("Num Allocations")
    axs[3].set_yscale("log")

    # --- Plot labour allocations ---
    lab_alloc_sizes = [M[t]["Lp"] for t in times]
    axs[4].plot(times, lab_alloc_sizes, marker='x')
    axs[4].set_title("Labour allocations (number)")
    axs[4].set_xlabel("Time")
    axs[4].set_ylabel("Num Allocations")

    # --- Plot rewards ---
    if rewards.shape[1] > 10:
        avg_rewards = np.mean(rewards, axis=1)
        std_rewards = np.std(rewards, axis=1)
        axs[5].plot(times, avg_rewards)
        axs[5].fill_between(times, avg_rewards - std_rewards, avg_rewards + std_rewards, color='blue', alpha=0.2, label='±1 Std Dev')
    else:
        axs[5].plot(times, rewards)
    
    axs[5].set_title("Rewards per agent or mean")
    axs[5].set_xlabel("Time")
    axs[5].set_ylabel("Reward")
    axs[5].set_yscale("log")

    # --- Plot gini ---
    gini_coeffs = np.array([gini(M[t]["C"]) for t in times])
    axs[6].plot(times, gini_coeffs, color='red')
    axs[6].set_title("Gini Coefficient (Capital Inequality)")
    axs[6].set_xlabel("Time")
    axs[6].set_ylabel("Gini")

    axs[7].plot(times, elasticities)
    axs[7].set_title("Process Elasticities")
    axs[7].set_xlabel("Time")
    axs[7].set_ylabel("e")

    axs[8].plot(times, wants)
    axs[8].set_title("Agent Wants")
    axs[8].set_xlabel("Time")
    axs[8].set_ylabel("W")

    axs[9].plot(times, n_labourers, color='red')
    axs[9].plot(times, n_capitalists, color='blue')
    axs[9].set_xlabel("Time")

    plt.tight_layout()
    if save:
        plt.savefig("last_run.png")
    plt.show()
