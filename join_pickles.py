import pandas as pd
import re
import os
from tqdm import tqdm
import pickle
from collections import defaultdict
import copy
import numpy as np
from preprocess_data import process_dictionary, process_frame


def all2df(dir_addr):
    files = []
    for file_name in tqdm(os.listdir(dir_addr)):
        if re.match('.+\.pkl', file_name):
            path = dir_addr + str(file_name)
            with open(path, "rb") as file:
                file = pickle.load(file)
            if file is not None:
                files.append(file)

    return files


def aggregate_files(files):
    print("aggregating...")

    D = defaultdict(list)
    df_frames = []

    for file_list in tqdm(files):
        for frame in file_list:
            new_metrics = process_frame(frame)
            frame.update(new_metrics)

            frame["p_elasticities"] = np.mean(frame["p_elasticities"])
            frame["Y"] = np.mean(frame["Y"])
            frame["Yopt"] = np.max(frame["Yopt"])
            frame["Ystd"] = np.mean(frame["Ystd"])
            frame["Ymedian"] = np.median(frame["Ymedian"])
            frame["Ymax"] = np.max(frame["Ymax"])
            df_frames.append(frame)

    df = pd.DataFrame(df_frames)

    # df = df.groupby(["alpha", "epsilon", "gamma", "n_agents", "n_processes"]).mean()

    # df.reset_index(inplace=True)

    print("length of merge", len(df))

    # add any other things to be calculated on the merged dataframe

    return df


def main(directory, save_directory):
    final_df = all2df(directory)
    final_df = aggregate_files(final_df)
    final_df.to_csv(save_directory + "process_experiments.csv")
    return final_df


if __name__ == "__main__":
    directory = "/cluster/work/coss/ccarissimo/capital_labour_processes/dataframes/"
    save_directory = "/cluster/work/coss/ccarissimo/capital_labour_processes/"

    files = all2df(directory)
    final_df = aggregate_files(files)
    final_df.to_csv(save_directory + "process_experiments.csv")

