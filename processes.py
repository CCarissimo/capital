import numpy as np


def production(multiplier, elasticity, capital, labour):
    return multiplier*capital**(elasticity)*labour**(1-elasticity)


def redistribution(Y, beta, capital_allocation, labour_allocation, n_labourers):


    # wage = min(10 * (1 - beta), to_labourers / n_labourers)
    # remainder = to_labourers - n_labourers * wage    
    # to_capitalists += remainder

    # shortcut
    if capital_allocation == 0 or labour_allocation == 0:
        return [.0, .0]
    else:
        to_capitalists = beta * (labour_allocation / capital_allocation) ** (1 - beta)
        to_labourers = (1 - beta) * (capital_allocation / labour_allocation) ** beta

        capital_return = to_capitalists
        labour_return = to_labourers
        return [labour_return, capital_return]


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = np.asarray(array).flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)  # Make all values non-negative
    array += 1e-10  # Avoid division by zero if all values are 0

    array = np.sort(array)
    n = array.shape[0]
    cumulative = np.cumsum(array)
    gini_coef = (2 * np.sum((np.arange(1, n + 1)) * array)) / (n * cumulative[-1]) - (n + 1) / n
    return gini_coef


def evolve_processes(elasticities, capital_allocations):
    not_funded_processes = np.nonzero(capital_allocations == 0)[0]
    # random_elasticity = np.random.random(len(not_funded_processes))
    random_mutation = np.random.normal(0, 0.1, size=len(not_funded_processes))
    elasticities[not_funded_processes] += random_mutation
    return np.clip(elasticities, 0, 1), not_funded_processes


def theoretical_max_production(mult, elasticity, max_labour):
    return np.exp(np.log(mult*max_labour**(1-elasticity))/(1-elasticity))
