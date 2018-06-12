import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('bmh')

# default plot width, can change module-wise from outside
plot_width = 12


def _def_figsize(ratio=0.6):
    """Return default figure size given y/x ratio"""
    return (plot_width, plot_width*ratio)


def _read_csv(file_name, filter_count=True, bad_visits=None):
    """Read data from CSV file
    """
    ds = pd.read_csv(file_name, header=0, index_col='visit')
    dscount = None
    if filter_count:
        # filter out records doing COUNT(*) which takes long time
        dscount = ds[ds.src_count.notnull()]
        ds = ds[ds.src_count.isnull()]
    if bad_visits:
        ds.drop(bad_visits, inplace=True)
    ds = ds.fillna(0, axis=1)
    return ds, dscount


def do_plot(ds, title, y, figsize=None):
    """Wrapper for Dataset.plot() with useful defaults"""
    figsize = figsize or _def_figsize()
    ds.plot(y=y, style=['o', 'r+'], title=title, figsize=figsize)


def do_boxplot(ds, by_col_name, columns, title='', bin=100, figsize=None):
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
    """

    figsize = figsize or _def_figsize(0.4)

    meanprops = dict(marker='s', markeredgecolor='black', markerfacecolor='red')
    for col in columns:
        ds.boxplot(col, by=by_col_name, figsize=figsize, showmeans=True,
                   meanprops=meanprops, sym='x', whis='range')
        x = ds.index
        y = ds[col]
        try:
            x = x/bin
            p = np.polyfit(x, y, 1)
            y = x*p[0] + p[1]
            label = "fit: {:.3f} + {:.3f}*visit/1000".format(p[1], p[0])
            # x + 1 is needed because boxplot draws in strange coordinates
            plt.plot(x+1, y, "--g", label=label)
            plt.legend()
        except:
            pass

        plt.title("")
        plt.suptitle(title + ": " + col)

def do_plots(file_name, title, bin=100, filter_count=True, bad_visits=None, 
             what=('scatter', 'counts', 'box'), time='visit_real'):
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
    """
    ds, dscount = _read_csv(file_name,  filter_count=filter_count, bad_visits=bad_visits)
    
    # do "scatter" plot
    if 'scatter' in what:
        do_plot(ds, title, y=[time])
    if 'counts' in what:
        do_plot(dscount, title, y=['obj_count', 'src_count'])
    
    # box plots
    if 'box' in what:
        col_name = 'visit/' + str(bin)
        ds[col_name] = np.array(ds.index/bin, dtype=np.int64)
        do_boxplot(ds, col_name, [time], bin=bin, title=title)

    return ds

def do_plots_all(file_name, title, bin=100, filter_count=True):
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
    """
    ds, dscount = _read_csv(file_name,  filter_count=filter_count)

    ds['obj_store'] = ds['obj_last_delete_real'] + ds['obj_last_insert_real'] + ds['obj_insert_real']
    
    # do "scatter" plot
    do_plot(ds, title, y=['select_real', 'store_real'])

    # box plots
    col_name = 'visit/' + str(bin)
    ds[col_name] = np.array(ds.index/bin, dtype=np.int64)
    do_boxplot(ds, col_name, ['select_real', 'store_real'], bin=bin, title=title)
    do_boxplot(ds, col_name, ['obj_select_real', 'obj_last_delete_real', 'obj_last_insert_real',
                              'obj_insert_real', 'obj_store', 'src_select_real', 'src_insert_real',
                              'fsrc_select_real', 'fsrc_insert_real'],
               bin=bin, title=title, figsize=_def_figsize(.35))
    return ds
