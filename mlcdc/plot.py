import matplotlib.pyplot as plt
import seaborn as sns

def histoscatter(x, y, ax=None, scatter_kw=None, hist_kw=None):
    """make a bivariate histogram, backed by a scatter plot for data below a threshold in terms of amount of data

    Args:
        x, y (array_like): the data making up x and y axes, respectively
        ax (:obj:`matplotlib.Axes`): axis object, created if not supplied
        scatter_kw (dict, optional): optional arguments passed to :func:`seaborn.scatterplot`
        hist_kw (dict, optional): optional arguments passed to :func:`seaborn.histplot`

    Returns:
        ax (:obj:`matplotlib.Axes`): the plotting axis
    """

    if ax is None:
        _, ax = plt.subplots()

    scatter_defaults = {"s":5, "color": "0.05"}
    if scatter_kw is None:
        scatter_kw = scatter_defaults
    else:
        for key,val in scatter_defaults.items():
            if key not in scatter_kw:
                scatter_kw[key] = val

    hist_defaults = {"bins":100, "cmap": "plasma", "pthresh":0.05}
    if hist_kw is None:
        hist_kw = hist_defaults
    else:
        for key,val in hist_defaults.items():
            if key not in hist_kw:
                hist_kw[key] = val

    sns.scatterplot(x=x, y=y, ax=ax, **scatter_kw)
    sns.histplot(x=x, y=y, ax=ax, **hist_kw)
    ax.plot(x, x, color=".8", alpha=.8)

    return ax
