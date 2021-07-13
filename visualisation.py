import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("paper")

def heatmap(X, save_path=None, **kwargs):
    """
    Plots simple heatmap by creating a matrix from the dataframe X
    """
    # set sequential colour palette to avoid confusion
    palette = sns.color_palette("viridis", as_cmap=True)
    sns.heatmap(data=X, center=0, cmap=palette, **kwargs)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()