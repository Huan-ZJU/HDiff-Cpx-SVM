import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

if __name__ == '__main__':
    new_adv_data = pd.read_excel("./corr_possamples.xlsx")
    print('head:', new_adv_data.head(), '\nShape:', new_adv_data.shape)
    print(new_adv_data.describe())
    print(new_adv_data[new_adv_data.isnull()].count())

    new_adv_data.dropna(inplace=True)
    arr = new_adv_data.corr()

    fig, ax = plt.subplots(figsize=(9, 9))
    sns.heatmap(pd.DataFrame(np.round(arr, 2)),
                annot=True, vmax=1, vmin=-1, xticklabels=True, yticklabels=True, cmap="RdYlBu_r")
    plt.savefig('corr.jpg', dpi=1200, format='jpg')
