import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.gridspec import GridSpec


# Function to analyze the data
def analyze_capital_accumulation(data_dict, capitals, wants):
    """
    Analyze capital accumulation over time in relation to agents' wants

    Parameters:
    data_dict: Dictionary with time steps as keys and results as values
    capitals: Initial capital endowment for each agent
    wants: Fixed costs/requirements for each agent
    """
    n_agents = len(wants)
    time_steps = sorted(list(data_dict.keys()))

    # Extract capital over time for each agent - vectorized approach
    capital_over_time = np.array([data_dict[t]["C"] for t in time_steps])

    # 1. Capital accumulation correlation with wants
    analyze_capital_wants_correlation(capital_over_time, wants, time_steps)

    # 2. Distribution shifts of capital over time
    analyze_capital_distribution_shifts(capital_over_time, time_steps)

    # 3. Rate of capital accumulation
    analyze_capital_growth_rate(capital_over_time, wants, time_steps)


def analyze_capital_wants_correlation(capital_over_time, wants, time_steps):
    """Analyze correlation between agents' wants and their capital accumulation"""
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2)

    # 1a. Calculate correlations vectorized
    # Create a matrix of correlations and p-values
    correlations = np.zeros(len(time_steps))
    p_values = np.zeros(len(time_steps))

    # Vectorized correlation calculation
    for t in range(len(time_steps)):
        corr, p_value = stats.pearsonr(wants, capital_over_time[t])
        correlations[t] = corr
        p_values[t] = p_value

    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(time_steps, correlations, marker='o', linestyle='-')
    ax1.set_title('Correlation between Wants and Capital over Time')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Pearson Correlation Coefficient')
    ax1.grid(True, alpha=0.3)

    # Use boolean indexing for significant correlations
    sig_mask = p_values < 0.05
    ax1.scatter(np.array(time_steps)[sig_mask], correlations[sig_mask],
                color='red', s=50, label='p < 0.05')
    ax1.legend()

    # 1b. Scatter plot of final capital vs wants
    ax2 = plt.subplot(gs[0, 1])
    final_capital = capital_over_time[-1]
    ax2.scatter(wants, final_capital, alpha=0.7)
    ax2.set_title(f'Final Capital vs Wants (Time Step {time_steps[-1]})')
    ax2.set_xlabel('Wants (Fixed Costs)')
    ax2.set_ylabel('Final Capital')

    # Add regression line
    slope, intercept, r_value, p_value, _ = stats.linregress(wants, final_capital)
    sorted_wants = np.sort(wants)
    ax2.plot(sorted_wants, intercept + slope * sorted_wants, 'r',
             label=f'R² = {r_value ** 2:.3f}, p = {p_value:.3f}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 1c. Heatmap: Capital over time by wants
    ax3 = plt.subplot(gs[1, :])

    # Sort agents by their wants - vectorized approach
    sorted_indices = np.argsort(wants)
    sorted_wants = np.array(wants)[sorted_indices]
    sorted_capital = capital_over_time[:, sorted_indices]

    # Prepare data for heatmap
    wants_categories = pd.qcut(sorted_wants, 10, labels=False)  # Group wants into deciles

    # Create empty array for grouped data
    groups = np.unique(wants_categories)
    grouped_data = np.zeros((len(time_steps), len(groups)))

    # Vectorized grouping by category
    for i, group in enumerate(groups):
        group_mask = wants_categories == group
        grouped_data[:, i] = np.mean(sorted_capital[:, group_mask], axis=1)

    # Plot heatmap
    sns.heatmap(grouped_data, cmap='viridis', ax=ax3)
    ax3.set_title('Average Capital over Time by Wants Decile')
    ax3.set_xlabel('Wants Decile (Low to High)')
    ax3.set_ylabel('Time Step')

    plt.tight_layout()
    plt.savefig('capital_wants_correlation.png')
    plt.close()

    print("Capital-wants correlation analysis completed.")


def analyze_capital_distribution_shifts(capital_over_time, time_steps):
    """Analyze how the distribution of capital changes over time"""
    num_plots = min(5, len(time_steps))
    selected_steps = np.linspace(0, len(time_steps) - 1, num_plots, dtype=int)

    plt.figure(figsize=(15, 10))

    # 2a. Distribution histograms at selected time steps
    for i, step_idx in enumerate(selected_steps):
        plt.subplot(2, 3, i + 1)
        sns.histplot(capital_over_time[step_idx], kde=True)
        plt.title(f'Capital Distribution at t={time_steps[step_idx]}')
        plt.xlabel('Capital')
        plt.ylabel('Frequency')

    # 2b. Calculate distribution metrics - vectorized approach
    # Gini coefficient calculation
    sorted_capitals = np.sort(capital_over_time, axis=1)
    n = sorted_capitals.shape[1]
    indices = np.arange(1, n + 1)

    # Calculate Gini coefficient using vectorized operations
    numerator = 2 * np.sum(indices * sorted_capitals, axis=1)
    denominator = n * np.sum(sorted_capitals, axis=1)
    gini_coeffs = (numerator / denominator) - (n + 1) / n

    # Other metrics - vectorized
    mean_vals = np.mean(capital_over_time, axis=1)
    median_vals = np.median(capital_over_time, axis=1)
    std_vals = np.std(capital_over_time, axis=1)

    # Plot metrics
    plt.subplot(2, 3, 6)
    plt.plot(time_steps, gini_coeffs, label='Gini Coefficient', marker='o')
    plt.plot(time_steps, std_vals / mean_vals, label='Coeff of Variation', marker='s')
    plt.title('Capital Inequality Metrics over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('capital_distribution_shifts.png')
    plt.close()

    # 2c. Calculate wealth percentiles - vectorized
    plt.figure(figsize=(10, 6))
    percentiles = [10, 25, 50, 75, 90]

    # Calculate all percentiles at once
    percentile_values = np.percentile(capital_over_time, percentiles, axis=1).T

    # Plot all percentiles
    for i, p in enumerate(percentiles):
        plt.plot(time_steps, percentile_values[:, i], label=f'{p}th Percentile', marker='o')

    plt.title('Capital Percentiles over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Capital Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('capital_percentiles.png')
    plt.close()

    print("Capital distribution shift analysis completed.")


def analyze_capital_growth_rate(capital_over_time, wants, time_steps):
    """Analyze rate of capital accumulation and its relationship with wants"""
    # Calculate growth rates - vectorized
    # Avoid division by zero with maximum
    denominator = np.maximum(capital_over_time[:-1], 1e-10)
    growth_rates = (capital_over_time[1:] - capital_over_time[:-1]) / denominator

    # Average growth rate per agent - vectorized
    avg_growth_rate = np.mean(growth_rates, axis=0)

    plt.figure(figsize=(15, 10))

    # 3a. Average growth rate vs wants
    plt.subplot(2, 2, 1)
    plt.scatter(wants, avg_growth_rate, alpha=0.7)
    plt.title('Average Growth Rate vs Wants')
    plt.xlabel('Wants (Fixed Costs)')
    plt.ylabel('Average Growth Rate')
    plt.grid(True, alpha=0.3)

    # Add regression line
    slope, intercept, r_value, p_value, _ = stats.linregress(wants, avg_growth_rate)
    sorted_wants = np.sort(wants)
    plt.plot(sorted_wants, intercept + slope * sorted_wants, 'r',
             label=f'R² = {r_value ** 2:.3f}, p = {p_value:.3f}')
    plt.legend()

    # 3b. Growth rate over time by wants category
    plt.subplot(2, 2, 2)
    # Group agents by wants into quartiles
    wants_quartiles = pd.qcut(wants, 4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    quartile_labels = np.sort(np.unique(wants_quartiles))

    # Prepare for vectorized quartile analysis
    quartile_growth_rates = []

    for quartile in quartile_labels:
        indices = np.where(wants_quartiles == quartile)[0]
        mean_growth = np.mean(growth_rates[:, indices], axis=1)
        quartile_growth_rates.append(mean_growth)
        plt.plot(time_steps[1:], mean_growth, marker='o', label=f'Wants {quartile}')

    plt.title('Growth Rate over Time by Wants Quartile')
    plt.xlabel('Time Step')
    plt.ylabel('Average Growth Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3c. Growth rate volatility vs wants - vectorized
    growth_volatility = np.std(growth_rates, axis=0)

    plt.subplot(2, 2, 3)
    plt.scatter(wants, growth_volatility, alpha=0.7)
    plt.title('Growth Rate Volatility vs Wants')
    plt.xlabel('Wants (Fixed Costs)')
    plt.ylabel('Growth Rate Standard Deviation')
    plt.grid(True, alpha=0.3)

    # Add regression line
    slope, intercept, r_value, p_value, _ = stats.linregress(wants, growth_volatility)
    plt.plot(sorted_wants, intercept + slope * sorted_wants, 'r',
             label=f'R² = {r_value ** 2:.3f}, p = {p_value:.3f}')
    plt.legend()

    # 3d. Cumulative growth by wants category - vectorized
    plt.subplot(2, 2, 4)
    denominator = np.maximum(capital_over_time[0], 1e-10)
    cumulative_growth = capital_over_time[-1] / denominator

    plt.scatter(wants, cumulative_growth, alpha=0.7)
    plt.title('Cumulative Capital Growth vs Wants')
    plt.xlabel('Wants (Fixed Costs)')
    plt.ylabel('Total Growth Factor (Final/Initial)')
    plt.grid(True, alpha=0.3)

    # Add regression line
    slope, intercept, r_value, p_value, _ = stats.linregress(wants, cumulative_growth)
    plt.plot(sorted_wants, intercept + slope * sorted_wants, 'r',
             label=f'R² = {r_value ** 2:.3f}, p = {p_value:.3f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig('capital_growth_analysis.png')
    plt.close()

    print("Capital growth rate analysis completed.")


# Example of how to use the analysis function
def run_analysis(data_dict, capitals, wants):
    """Run all analyses on the provided data"""
    print("Starting capital accumulation analysis...")
    analyze_capital_accumulation(data_dict, capitals, wants)
    print("Analysis complete! Check the generated visualization files.")