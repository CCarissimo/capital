import numpy as np
import pandas as pd


def process_frame(entry):
    n_processes = entry["n_processes"]
    p_elasticities = np.array(entry['p_elasticities'])
    Y = np.array(entry['Y'])
    Ymax = np.array(entry['Ymax'])
    Yopt = np.array(entry['Yopt'])

    if np.where(p_elasticities < 0.1, 1, 0).sum() > 0:
        print(entry)

    # Compute metrics
    p_at_max_Y = p_elasticities[np.argmax(Y)]
    p_at_max_Ymax = p_elasticities[np.argmax(Ymax)]
    Y_over_Yopt = Y / Yopt
    p_at_max_YdivYopt = p_elasticities[np.argmax(Y_over_Yopt)]
    frac_below_mean_Y = np.mean(Y < np.mean(Y))

    results = {
        'n_processes': n_processes,
        'p_at_max_Y': p_at_max_Y,
        'p_at_max_Ymax': p_at_max_Ymax,
        'p_at_max_YdivYopt': p_at_max_YdivYopt,
        'frac_below_mean_Y': frac_below_mean_Y,
    }

    return results


def process_dictionary(D):
    results = []

    for n_processes, dict_list in D.items():
        for entry in dict_list:
            frame = process_frame(entry)

            # Store results
            results.append(frame)

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)
    return df_results
