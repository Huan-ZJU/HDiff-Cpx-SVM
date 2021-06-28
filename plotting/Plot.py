import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot(table, n_C, n_gamma, ):
    array = np.array(table)

    # create image
    image = np.zeros((n_C, n_gamma))
    for i in range(n_C):
        for j in range(1, n_gamma):
            image[i, j] = array[i * n_gamma + j][2]

    for i in range(n_C):
        image[i, 0] = np.min(image[:, 1:100])

    image2 = np.zeros((n_C, n_gamma))
    for i in range(n_C):
        for j in range(n_gamma):
            image2[i, j] = array[i * n_gamma + j][3]

    plt.subplots(figsize=(6, 5))
    ax = sns.heatmap(image2, cmap='jet', linewidths=0, yticklabels=False, xticklabels=False, )
    plt.savefig('v.jpg', dpi=1200, format='jpg')

    plt.subplots(figsize=(6, 5))
    ax = sns.heatmap(image, cmap='jet', linewidths=0, yticklabels=False, xticklabels=False, )
    c_bar = ax.collections[0].colorbar
    c_bar.set_ticks([0, 1, ])
    c_bar.set_ticklabels(['0', '1', ])
    plt.savefig('s.jpg', dpi=1200, format='jpg')

    print("OK!")
