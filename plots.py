import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# default plot width, can change module-wise from outside
plot_width = 12

DEFAULT_PLOTS = ['select_real', 'store_real', 'obj_select_real', 'obj_last_delete_real',
                 'obj_last_insert_real', 'obj_insert_real', 'obj_store',
                 'src_select_real', 'src_insert_real', 'fsrc_select_real', 'fsrc_insert_real']

DEFAULT_FITS = [
    ['obj_select_real', 'src_select_real', 'fsrc_select_real', 'select_real'],
    ['obj_select_cpu',  'src_select_cpu',  'fsrc_select_cpu'],
    ['obj_insert_real', 'src_insert_real', 'fsrc_insert_real', 'store_real', 'obj_last_insert_real', ],
    ['obj_insert_cpu',  'src_insert_cpu',  'fsrc_insert_cpu'],
    ['store_cpu'],
]

def _def_figsize(ratio=0.6):
    """Return default figure size given y/x ratio"""
    return (plot_width, plot_width*ratio)


def _read_csv(file_name, filter_count=True, bad_visits=None, fix_select_real=False):
    """Read data from CSV file
    """
    ds = pd.read_csv(file_name, header=0, index_col='visit')
    if bad_visits:
        ds.drop(bad_visits, inplace=True)
    if fix_select_real:
        ds.select_real *= 3
    dscount = None
    if filter_count:
        # filter out records doing COUNT(*) which takes long time
        dscount = ds[ds.src_count.notnull()]
        ds = ds[ds.src_count.isnull()]
    ds = ds.fillna(0, axis=1)
    return ds, dscount


def do_plot(ds, title, y, figsize=None):
    """Wrapper for Dataset.plot() with useful defaults"""
    figsize = figsize or _def_figsize()
    ds.plot(y=y, style=['.', '.'], title=title, figsize=figsize)


def do_boxplot(ds, by_col_name, columns, title='', bin=100, figsize=None, whis="range"):
    """Make a boxplot by grouping data based on some column

    Parameters
    ----------
    dataset : Dataset
    by_col_name : str
        Column name used for data grouping
    columns : list
        Columns to plot, each is plotted on a separate figure
    title : str
        Plot title, column name will be appended to it
    bin : int
        Grouping bin width (number of visits per bin)
    figsize : tuple
    whis : 
        option for boxplot
    """

    figsize = figsize or _def_figsize(0.4)

    meanprops = dict(marker='s', markeredgecolor='black', markerfacecolor='red')
    for col in columns:
        pos = ds[by_col_name].unique() * (float(bin)/1000)
        width = float(bin)/2000
        axes = ds.boxplot(col, by=by_col_name, figsize=figsize, showmeans=True,
                          positions=pos, widths=width, meanprops=meanprops, sym='x', whis=whis)
        axes.set_xticks(pos)
        axes.set_xticklabels(["{:.4g}".format(p) for p in pos])
        x = ds.index
        y = ds[col]
        try:
            # fit with a straight line
            p = np.polyfit(x, y, 1)
            label = "fit: {:.3f} + {:.3f}*visit/1000".format(p[1], p[0]*1000)
            # plot the line, axes are in the space of by_col_name
            y = pos*(p[0]*1000) + p[1]
            axes.plot(pos, y, "--g", label=label)
            plt.legend()
        except:
            raise

        plt.title("")
        plt.suptitle(title + ": " + col)

def do_plots(file_name, title, bin=100, filter_count=True, bad_visits=None,
             what=('scatter', 'counts', 'box', 'fit'),
             time='visit_real', fix_select_real=False, whis="range"):
    """Make bunch of plots for one time column.

    This method is typically used for multi-process configurations where only
    the per-visit times make sense.

    Parameters
    ----------
    file_name : str
        Name of the CSV file with data
    title : str
        Title for plots
    bin : int
        Grouping bin width (number of visits per bin) for box plots
    filter_count : bool
        If true then do not include visits where table sizes were queried
    bad_visits : list
        List of visits to exclude from plots
    what : sequence of str
        Defines what plots to produce, can include 'scatter', 'counts', 'box'
    time : str
        Name of the column with time data
    fix_select_real : bool
        If True then "select_real" column will be multiplied by 3
    """
    ds, dscount = _read_csv(file_name,  filter_count=filter_count, bad_visits=bad_visits, fix_select_real=fix_select_real)

    # do "scatter" plot
    if 'scatter' in what:
        do_plot(ds, title, y=[time])
    if 'counts' in what:
        do_plot(dscount, title, y=['obj_count', 'src_count'])

    # box plots
    if 'box' in what:
        col_name = 'visit/1000'
        ds[col_name] = np.array((ds.index+bin/2)/bin, dtype=np.int64)
        do_boxplot(ds, col_name, [time], bin=bin, title=title, whis=whis)

    # box plots
    if 'fit' in what:
        plot_fit_times(ds, [time], title=title)

    return ds

def do_plots_all(input, title, bin=100, filter_count=True, plots=None, fits=None,
                 fit_mode="poly", bad_visits=None, fix_select_real=False, whis="range",
                 fit_nbins=50):
    """Make bunch of plots for one time column.

    This method is typically used for multi-process configurations where only
    the per-visit times make sense.

    Parameters
    ----------
    input : `str` or `pandas.DataFrame`
        Name of the CSV file with data or pre-filled DataFrame
    title : str
        Title for plots
    bin : int
        Grouping bin width (number of visits per bin) for box plots
    filter_count : bool
        If true then do not include visits where table sizes were queried
    plots : sequence of str
        Names of the variables to plot, if None then default list is used
    fits : sequence of sequence
        Groups of columns used for time plots with fits
    fit_mode : str or dict
        Dict maps column name to fit mode. Fit mode can be one of 'lin', 'poly',
        or None. If there is no mapping for a column then 'poly' is used.
    bad_visits : list
        List of visits to exclude from plots
    fix_select_real : bool
        If True then "select_real" column will be multiplied by 3
    fit_nbins : int
        Number of bins for fit plots.
    """
    if isinstance(input, pd.DataFrame):
        ds = input
    else:
        ds, dscount = _read_csv(input,  filter_count=filter_count, bad_visits=bad_visits, fix_select_real=fix_select_real)

    ds['obj_store'] = ds['obj_last_delete_real'] + ds['obj_last_insert_real'] + ds['obj_insert_real']

    # do "scatter" plot
    do_plot(ds, title, y=['select_real', 'store_real'])
    plot_fit_times(ds, ['select_real', 'store_real'], title=title, fit_mode=fit_mode, nbins=fit_nbins)

    # box plots
    col_name = 'visit/1000'
    ds[col_name] = np.array((ds.index+bin/2)/bin, dtype=np.int64)
    if plots is None:
        plots = DEFAULT_PLOTS
    do_boxplot(ds, col_name, plots, bin=bin, title=title, figsize=_def_figsize(.35), whis=whis)
    if fits is None:
        fits = DEFAULT_FITS
    for fit in fits:
        plot_fit_times(ds, fit, title=title, fit_mode=fit_mode, nbins=fit_nbins)
    return ds

def fit_times(ds, column, mode='poly'):
    """Fit column vs visit.

    Parameteres
    -----------
    ds : `pandas.DataFrame`
    column : `str`
        Names of the column with time data
    mode : str
        "lin" - linear fit (y = p0*x), "poly" - p1 fit (y = p0*x + p1)

    Returns
    -------
    ax
    """

    def fun_lin(x, visits, data):
        # print(x)
        visits = visits.to_numpy()
        data = data.to_numpy()
        p0, = x
        res = p0*visits - data
        return res

    def fun_poly(x, visits, data):
        # print(x)
        visits = visits.to_numpy()
        data = data.to_numpy()
        p0, p1 = x
        res = p0*visits + p1 - data
        return res

    visits = pd.Series(ds.index)
    params = {}
    coldata = ds[column]
    if mode == 'poly':
        ores = least_squares(fun_poly, (0., 0.), args=(visits, coldata))
        p = ores.x
    elif mode == 'lin':
        ores = least_squares(fun_lin, (0.,), args=(visits, coldata))
        p = ores.x
    return p


def plot_fit_times(ds, columns, fit_mode="poly", nbins=75, ax=None, figsize=None, title="", ylabel="Time, sec"):
    """Plot a fit of single column vs visit

    Parameteres
    -----------
    ds : `pandas.DataFrame`
    columns : list of str
        Names of the columns with time data
    fit_mode : str or dict
        Dict maps column name to fit mode. Fit mode can be one of 'lin', 'poly',
        or None. If there is no mapping for a column then 'poly' is used.
    nbins : int
        Number of bins for plotting
    ax : Axes
    figsize : tuple, optional
    ylabel : str
        Label for Y axis

    Returns
    -------
    ax
    """
    def fun_lin(x, visits, data):
        # print(x)
        visits = visits.to_numpy()
        data = data.to_numpy()
        p0, = x
        res = p0*visits - data
        return res

    def fun_poly(x, visits, data):
        # print(x)
        visits = visits.to_numpy()
        data = data.to_numpy()
        p0, p1 = x
        res = p0*visits + p1 - data
        return res

    if ax is None:
        figsize = figsize or _def_figsize(0.6)
        f, ax = plt.subplots(figsize=figsize)
    if title:
        plt.title(title)

    visits = pd.Series(ds.index)
    for col in columns:
        coldata = ds[col]

        if hasattr(fit_mode, "get"):
            mode = fit_mode.get(col, 'poly')
        else:
            mode = fit_mode

        if mode == 'poly':
            ores = least_squares(fun_poly, (0., 0.), args=(visits, coldata))
            p = ores.x
            label = "{}: {:.4f} + {:.4f}*visit/1000".format(col, p[1], p[0]*1000)
            fit_reg = True
        elif mode == 'lin':
            ores = least_squares(fun_lin, (0.,), args=(visits, coldata))
            p = ores.x
            label = "{}: {:.4f}*visit/1000".format(col, p[0]*1000)
            fit_reg = True
        else:
            label = col
            fit_reg = False

        sns.regplot(x=visits, y=coldata, x_bins=nbins, line_kws=dict(linestyle="--"),
                    label=label, ax=ax, fit_reg=fit_reg)

    if ylabel:
        ax.set_ylabel("Time, sec")
    ax.legend(loc="best")

    return ax

def plot_fraction_above(ds, by_col_name, bin, column="visit_real", title='',
                        threshold=10, figsize=None, ax=None):
    """Plot fraction of the events where time exceeds threshold.
    """
    if ax is None:
        figsize = figsize or _def_figsize(0.4)
        f, ax = plt.subplots(figsize=figsize)
    if title:
        plt.title(title)

    ds["fraction_above"] = ds[column] > threshold
    frac = ds[[by_col_name, "fraction_above"]].groupby(by_col_name).agg("mean")
    x = frac.index * (float(bin)/1000)
    ax.plot(x, frac.fraction_above, marker="*", label="{} > {:g}".format(column, threshold))
    ax.set_xlabel("visit/1000")
    ax.set_ylabel("Fraction of visits")
    ax.legend(loc="best")

    return ax
