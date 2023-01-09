import re
import threading
import warnings
from enum import Enum, auto
from typing import Optional, Union

import matplotlib
import numpy as np
from matplotlib import pyplot as plt, pylab
from matplotlib.backend_bases import FigureManagerBase
from pymoo.core.plot import Plot

from logger import fatal, warn
from utils_date import getNowStr
from utils_fs import createFolder, pathJoin
from utils_misc import getRunIdStr, runWithExpRetry, size


class PlotMode(Enum):
    BLOCKING_SHOW = auto()
    NON_BLOCKING_SHOW = auto()
    SAVE_TO_FILE = auto()
    DONT_PLOT = auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def getAll():
        return list(map(lambda c: c, PlotMode))


DEFAULT_PLOT_MODE = PlotMode.SAVE_TO_FILE
DEFAULT_BACKEND = None
MAIN_THREAD_FIGURE_MANAGER = None
SAVE_FILE_BACKEND = 'Agg'
SAVED_PLOTS_PATH = 'saved_plots'
FIGURE_EXTRA_WIDTH_RATIO_FOR_LEGEND = 1.1
FIGURE_WIDTH = 1920
FIGURE_HEIGHT = 1080
FIGURE_DPI = 150
FIGURE_LEGEND_X_ANCHOR = 1.0
FIGURE_LEGEND_Y_ANCHOR = 0.5

_saved_plots_counter = 0
_had_a_non_blocking = False

_current_backend = DEFAULT_BACKEND


def getDefaultBackend():
    global DEFAULT_BACKEND
    if DEFAULT_BACKEND is None:
        DEFAULT_BACKEND = matplotlib.get_backend()
    return DEFAULT_BACKEND


def setCurrentBackend(backend: str):
    global _current_backend
    _current_backend = backend


def getCurrentBackend() -> str:
    if _current_backend is None:
        setCurrentBackend(getDefaultBackend())
    return _current_backend


def getPlotColorFromIndex(idx: int, colours_to_avoid: Optional[Union[list, str]] = None) -> str:
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    # background = 'w'
    all_colours = ['b', 'g', 'r', 'c', 'y', 'tab:gray', 'tab:pink', 'tab:brown', 'tab:purple', 'tab:orange',
                   'chartreuse', 'm', 'cornflowerblue', 'darkviolet', 'crimson', 'fuchsia', 'salmon', 'indigo', 'k']
    if colours_to_avoid is not None and type(colours_to_avoid) is str:
        colours_to_avoid = [colours_to_avoid]
    available_colours = []
    for c in all_colours:
        if colours_to_avoid is None or c not in colours_to_avoid:
            available_colours.append(c)
    return available_colours[(idx % len(available_colours))]


def getFigureManager() -> Optional[FigureManagerBase]:
    global MAIN_THREAD_FIGURE_MANAGER
    return MAIN_THREAD_FIGURE_MANAGER


def resizeFigure(width: int = FIGURE_WIDTH, height: int = FIGURE_HEIGHT) -> None:
    getFigureManager().resize(width, height)


def getNextPlotFilepath(prefix: str = 'plot', label: str = '', plot_subdir: Optional[str] = None,
                        add_run_id: bool = True, counter_postfix: bool = False, datetime_postfix: bool = False) -> str:
    global _saved_plots_counter

    if prefix.strip() != '':
        prefix += '-'
    postfix = ''
    if counter_postfix:
        postfix += f'-{_saved_plots_counter}'
        _saved_plots_counter += 1
    if datetime_postfix:
        postfix += f'-{getNowStr(output_format="%d%m%Y%H%M%S")}'

    filename = f'{prefix}{f"{label}" if label else ""}{postfix}.png'
    if add_run_id:
        base_path = pathJoin(SAVED_PLOTS_PATH, getRunIdStr())
        createFolder(base_path)
    else:
        base_path = SAVED_PLOTS_PATH
    if plot_subdir is not None:
        base_path = pathJoin(base_path, plot_subdir)
        createFolder(base_path)
    filepath = pathJoin(base_path, filename)
    return filepath


def clearCurrentFigure() -> None:
    plt.clf()
    plt.cla()
    plt.close()  # delete the last and empty figure


def blockPlots() -> None:
    plt.show()


def maybeBlockPlots() -> None:
    if _had_a_non_blocking:
        blockPlots()


def _line(plot_data: Union[list, np.ndarray], plot_args: dict) -> bool:
    if 'style' in plot_args:
        style = plot_args.pop('style')
        plot_data.append(style)
    if size(plot_data) > 0 and size(plot_data[0]) > 0:
        plt.plot(*plot_data, **plot_args)
        return True
    return False


def _scatter(plot_data: Union[list, np.ndarray], plot_args: dict) -> bool:
    if size(plot_data) > 0 and size(plot_data[0]) > 0:
        plt.scatter(*plot_data, **plot_args)
        return True
    return False


def _imshow(plot_data: Union[list, np.ndarray], plot_args: dict) -> bool:
    if 'auto_range' in plot_args:
        plot_args.pop('auto_range')
        plot_data[0] = np.array(plot_data[0])
        plot_args['vmin'] = plot_data[0].min()
        plot_args['vmax'] = plot_data[0].max()
        if 'auto_range_factors' in plot_args:
            fmin, fmax = plot_args.pop('auto_range_factors')
            plot_args['vmin'] *= fmin
            plot_args['vmax'] *= fmax
    if size(plot_data) > 0 and size(plot_data[0]) > 0:
        plt.imshow(*plot_data, **plot_args)
        return True
    return False


def _colorbar(_: Union[list, np.ndarray], plot_args: dict) -> bool:
    plt.colorbar(**plot_args)
    return True


def _suptitle(plot_data: Union[list, np.ndarray], plot_args: dict) -> bool:
    plt.suptitle(*plot_data, **plot_args)
    return True


def _axhline(plot_data: Union[list, np.ndarray], plot_args: dict) -> bool:
    plt.axhline(*plot_data, **plot_args)
    return True


def _axvline(plot_data: Union[list, np.ndarray], plot_args: dict) -> bool:
    plt.axvline(*plot_data, **plot_args)
    return True


def _text(plot_data: Union[list, np.ndarray], plot_args: dict) -> bool:
    plt.text(*plot_data, **plot_args)
    return True


def maybeSetFigureManager():
    global MAIN_THREAD_FIGURE_MANAGER
    if MAIN_THREAD_FIGURE_MANAGER is None and threading.current_thread() is threading.main_thread():
        MAIN_THREAD_FIGURE_MANAGER = plt.get_current_fig_manager()


def getColorGradientsFromIndex(i: int, for_confusion_matrix: bool = False) -> str:
    if for_confusion_matrix:
        cmaps = ['bwr', 'seismic', 'Blues', 'OrRd']
    else:
        cmaps = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
                 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r',
                 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1',
                 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr',
                 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu',
                 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2',
                 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu',
                 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn',
                 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis',
                 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix',
                 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat',
                 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern',
                 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray',
                 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet',
                 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
                 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket',
                 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r',
                 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'viridis',
                 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']
    return cmaps[i % len(cmaps)]


def plot(plots: Union[tuple[str, list, dict], list[tuple[str, list, dict]]], mode: Optional[PlotMode] = None,
         title: Optional[str] = None, x_label: Optional[Union[str, tuple[str, dict]]] = None,
         y_label: Optional[Union[str, tuple[str, dict]]] = None, legend: Union[bool, str, dict] = False,
         legend_outside: Union[bool, float] = False, tight_layout: bool = True,
         resize: Union[bool, tuple[int, int]] = False,
         x_ticks: Optional[tuple[Union[list, np.ndarray, dict], ...]] = None,
         y_ticks: Optional[tuple[Union[list, np.ndarray, dict], ...]] = None, file_label: Optional[str] = None,
         subdir: Optional[str] = None, add_rid_subdir: bool = True, file_prefix: Union[str, bool] = 'plot',
         file_postfix: bool = True, file_datetime_postfix: bool = True, file_counter_postfix: bool = False) -> None:
    global _had_a_non_blocking
    current_backend = getCurrentBackend()
    default_backend = getDefaultBackend()

    maybeSetFigureManager()

    if mode is None:
        mode = DEFAULT_PLOT_MODE

    if mode == PlotMode.DONT_PLOT:
        return

    if mode in (PlotMode.BLOCKING_SHOW, PlotMode.NON_BLOCKING_SHOW):
        if current_backend != default_backend:
            matplotlib.use(default_backend)
            setCurrentBackend(default_backend)
    elif current_backend != SAVE_FILE_BACKEND:
        matplotlib.use(SAVE_FILE_BACKEND)
        setCurrentBackend(SAVE_FILE_BACKEND)
    if type(plots) is tuple:
        plots = [plots]
    plot_something = False
    for p in plots:
        plot_type = p[0].lower().strip()
        plot_data = p[1]
        plot_args = p[2]

        if plot_type in ('plot', 'line'):
            plot_something = _line(plot_data, plot_args) or plot_something
        elif plot_type == 'scatter':
            plot_something = _scatter(plot_data, plot_args) or plot_something
        elif plot_type in ('imshow', 'im'):
            plot_something = _imshow(plot_data, plot_args) or plot_something
        elif plot_type == 'colorbar':
            _colorbar(plot_data, plot_args)
        elif plot_type == 'suptitle':
            _suptitle(plot_data, plot_args)
        elif plot_type == 'axhline':
            _axhline(plot_data, plot_args)
        elif plot_type == 'axvline':
            _axvline(plot_data, plot_args)
        elif plot_type == 'text':
            _text(plot_data, plot_args)
        else:
            fatal(Exception(f'Invalid plot_type: `{plot_type}`'))

    if not plot_something:
        return

    if title is not None:
        plt.title(title)
        fig = pylab.gcf()
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title(title)
    if x_label is not None:
        if type(x_label) is tuple:
            plt.xlabel(*x_label[:-1], **x_label[-1])
        else:
            plt.xlabel(x_label)
    if y_label is not None:
        if type(y_label) is tuple:
            plt.ylabel(*y_label[:-1], **y_label[-1])
        else:
            plt.ylabel(y_label)
    if x_ticks is not None:
        plt.xticks(*x_ticks[:-1], **x_ticks[-1])
    if y_ticks is not None:
        plt.yticks(*y_ticks[:-1], **y_ticks[-1])
    if tight_layout or (type(legend_outside) is bool and legend_outside) or type(legend_outside) is float:
        warnings.filterwarnings("error")
        try:
            if not legend_outside:
                plt.tight_layout()
            else:
                width = FIGURE_EXTRA_WIDTH_RATIO_FOR_LEGEND
                if type(legend_outside) is float:
                    width += legend_outside
                plt.tight_layout(rect=[0, 0, width, 1])
        except Warning as e:
            warn(f'Tight Layout Issue: title: {title} | x_label: {x_label} | y_label: {y_label} | legend: {legend} | '
                 f'legend_outside: {legend_outside} | tight_layout: {tight_layout} | file_label: {file_label} | '
                 f'subdir: {subdir} | file_prefix: {file_prefix}')
            warn(e)
        warnings.resetwarnings()

    if (type(legend_outside) is bool and legend_outside) or type(legend_outside) is float:
        plt.legend(loc='center left', bbox_to_anchor=(FIGURE_LEGEND_X_ANCHOR, FIGURE_LEGEND_Y_ANCHOR))
    elif type(legend) is str:
        plt.legend(legend)
    elif type(legend) is dict:
        plt.legend(**legend)
    elif type(legend) is bool and legend:
        plt.legend()

    if (type(resize) is bool and resize) or type(resize) is tuple:
        if type(resize) is tuple:
            resizeFigure(resize[0], resize[1])
        else:
            resizeFigure()

    if mode in (PlotMode.BLOCKING_SHOW, PlotMode.NON_BLOCKING_SHOW):
        if mode == PlotMode.BLOCKING_SHOW:
            plt.show(block=True)
        else:
            plt.show(block=False)
            _had_a_non_blocking = True
            clearCurrentFigure()  # to clean up, when show not blocking or saving to file
            plt.figure(dpi=FIGURE_DPI)
    else:
        createFolder(SAVED_PLOTS_PATH)
        if type(file_prefix) is bool:
            if file_prefix:
                file_prefix = 'plot'
            else:
                file_prefix = ''
        if not file_postfix:
            file_counter_postfix = file_datetime_postfix = False
        if file_label is None:
            file_label = re.sub(r'\W+', '', title.replace(' ', '_').lower())
        filepath = getNextPlotFilepath(prefix=file_prefix, label=file_label, plot_subdir=subdir,
                                       add_run_id=add_rid_subdir, counter_postfix=file_counter_postfix,
                                       datetime_postfix=file_datetime_postfix)

        runWithExpRetry(f'SavePlot-{file_label}', plt.savefig, [filepath], dict(bbox_inches='tight', dpi=FIGURE_DPI),
                        3, raise_it=False)
        clearCurrentFigure()  # to clean up, when show not blocking or saving to file
        plt.figure(dpi=FIGURE_DPI)


def showOrSavePymooPlot(the_plot: Plot, label: str, subsubdir: Optional[str] = None, prefix: str = '') -> None:
    try:
        resizeFigure()
        plot_subdir = 'nas'
        if subsubdir is not None:
            pathJoin(plot_subdir, subsubdir)
        if DEFAULT_PLOT_MODE == PlotMode.SAVE_TO_FILE:
            filepath = getNextPlotFilepath(prefix=prefix, label=label, plot_subdir=plot_subdir,
                                           add_run_id=True, counter_postfix=False,
                                           datetime_postfix=False)
            the_plot.save(filepath, dpi=FIGURE_DPI)
        elif DEFAULT_PLOT_MODE in (PlotMode.BLOCKING_SHOW, PlotMode.NON_BLOCKING_SHOW):
            the_plot.show()
    except:
        del the_plot


def getCMap(cmap_name: str) -> object:
    return matplotlib.colormaps[cmap_name]


matplotlib.rcParams['figure.dpi'] = FIGURE_DPI
