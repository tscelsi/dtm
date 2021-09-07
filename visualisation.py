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
    plt.clf()
    palette = sns.color_palette("viridis", as_cmap=True)
    g = sns.heatmap(data=X, center=0, cmap=palette, **kwargs)
    if save_path:
        print("saving heatmap...")
        plt.savefig(save_path)
    # else:
    #     print("showing heatmap...")
    #     plt.show()
    return plt, g


def time_evolution_plot(df, filename, title=None, scale=1, save=True):
    """This function plots topic proportions over time. This function was adapted
    and is credited to the Muller-Hansen paper, the original code can be found in the
    following repository: https://github.com/mcc-apsis/coal-discourse.

    Args:
        df (pd.DataFrame): This is the dataframe that contains the data for plotting. The data should look
        similar to:
            topic_name          0         1        10  ...         7         8         9
            year                                       ...                              
            1997         1.388494  4.851258  0.682613  ...  2.151749  5.940288  5.549168
            1999         1.824987  4.308183  0.369598  ...  2.946666  4.397723  4.819061
            2000         2.007208  4.465947  0.954091  ...  7.107205  4.475813  5.143148
            2001         1.962062  5.138283  0.208519  ...  5.727354  4.850402  4.928038
            2002         2.296691  4.087264  1.247498  ...  5.597781  3.957320  5.775876
            2003         1.361571  3.489412  7.097156  ...  4.880698  3.832103  5.031273
            2004         2.264976  2.877810  4.056191  ...  3.253473  3.314512  4.896444
            2005         1.887321  3.466041  3.832519  ...  2.648234  4.212436  5.088535
            2006         1.456009  2.730801  2.910064  ...  3.306952  3.548342  5.672392
            2007         1.675358  2.575447  2.383468  ...  4.219317  3.666694  4.881267
            2008         1.786699  3.186896  1.782014  ...  2.834857  4.389405  6.141509
            2009         1.760088  3.462534  5.487852  ...  2.095825  3.013996  5.754901
        
            Where the columns represent the topics, each row a timestep and the cell values are the
            proportional relevance of a topic at a particular timestep.
            
        filename (str): Path which to save the plot figure to.
        title (str, optional): The title of the plot. Defaults to None.
        scale (int, optional): scale factor that dictates how fat the topic proportion representations are. 
            Defaults to 1.
        save (bool, optional): Whether or not to save the figure. Defaults to True.

    Returns:
        matplotlib plot object
    """
    sns.set_context("talk")
    sns.set_style("ticks")
    sns.set_style({'axes.spines.bottom': True,
                'axes.grid':True,
                'axes.spines.left': False,
                'axes.spines.right': False,
                'axes.spines.top': False,
                'ytick.left': False,
                'figure.facecolor':'w'})
    fig = plt.figure(figsize=(30, 1.7 * len(df.columns)))
    ax = fig.gca()

    plt.yticks([])
    x_domain = [x for x in range(1,len(df)+1)]
    x_labels = df.index.tolist()
    assert len(x_domain) == len(x_labels)
    plt.xticks(x_domain)
    ax.set_xticklabels(x_labels)
    max_val = scale * df.max().max() + 5

    for i, t in enumerate(reversed(df.columns)):
        plt.fill_between(x_domain, df[t] + i*max_val, i*max_val - df[t], label=t)
        plt.text(len(df) + 0.3, (i+0.) *max_val, t)

    plt.xlabel('Year')
    if title:
        plt.title(title)
    if save:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    return plt

def plot_word_ot(df, title, save_path=None):
    plt.clf()
    fig = plt.figure(figsize=(10,10))
    palette = sns.color_palette("viridis", as_cmap=True)
    sns.lineplot(data=df).set(title=title)
    if save_path:
        plt.savefig(save_path)
    return plt