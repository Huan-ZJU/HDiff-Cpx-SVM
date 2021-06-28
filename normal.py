import numpy as np
import pandas as pd


def normal(tonormalizedpath, filepath, stretch):
    pd_normal_data = pd.read_csv(tonormalizedpath)
    np_normal_data = pd_normal_data.values

    # normalization based on training data
    for i in range(np_normal_data.shape[0]):
        np_normal_data[i][2] = normal1(np_normal_data[i][2], 1.999, 1.698, stretch)
        np_normal_data[i][3] = normal1(np_normal_data[i][3], 0.117, 0.003, stretch)
        np_normal_data[i][4] = normal1(np_normal_data[i][4], 0.394, 0.012, stretch)
        np_normal_data[i][5] = normal1(np_normal_data[i][5], 0.05, 0, stretch)
        np_normal_data[i][6] = normal1(np_normal_data[i][6], 0.303, 0.093, stretch)
        np_normal_data[i][7] = normal1(np_normal_data[i][7], 1.023, 0.549, stretch)
        np_normal_data[i][8] = normal1(np_normal_data[i][8], 1.029, 0.626, stretch)
        np_normal_data[i][9] = normal1(np_normal_data[i][9], 0.011, 0, stretch)
        np_normal_data[i][10] = normal1(np_normal_data[i][10], 0.135, 0.007, stretch)
        np_normal_data[i][11] = normal1(np_normal_data[i][11], 856, 0.001, stretch)

    # delete previous data
    complete_set = pd_normal_data.drop(list(pd_normal_data.columns), axis=1)
    # insert normalized data
    complete_set[pd_normal_data.columns] = pd.DataFrame(np_normal_data, columns=pd_normal_data.columns)
    # output
    complete_set.to_csv(filepath, index=False)


def normal1(x, max_data, min_data, stretch):
    x = (x - min_data) / (max_data - min_data) * stretch
    return x
