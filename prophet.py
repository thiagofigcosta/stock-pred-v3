import math
import os
from typing import Union, Optional

import numpy as np

from hyperparameters import Hyperparameters
from logger import verbose, error, info, clean, getVerbose
from metrics import getKerasRegressionMetrics, getAllCustomMetrics, EPSILON
from plotter import FIGURE_DPI, plot, getPlotColorFromIndex, getColorGradientsFromIndex
from postprocessor import decodeWindowedPredictions, decodeWindowedLabels, computeMetricsAndGetValues, AggregationMethod
from preprocessor import ProcessedDataset, DatasetSplit
from prophet_enums import Optimizer, ActivationFunc
from transformer import getTransformedTickerFilepath
from utils_date import getNowStr
from utils_fs import createFolder, pathJoin, removeFileExtension, getBasename, pathExists, copyFile, deleteFile, \
    moveFile
from utils_misc import numpyToListRecursive, listToChunks, getNumericTypes, getCpuCount, getRunIdStr, runWithExpRetry
from utils_persistance import saveJson, loadJson, loadObj

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # DISABLE TENSORFLOW WARNING
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, History
from keras.layers import LSTM, Dropout, Dense, LeakyReLU, Bidirectional
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import Callback
from keras.regularizers import L1L2

MODELS_DIR = 'models'
CHECKPOINT_SUBDIR = 'checkpoint'
ARCHITECTURE_SUBDIR = 'architecture'
HISTORY_SUBDIR = 'history'
METRICS_SUBDIR = 'metrics'
PROPHET_DIR = 'prophets'


class CustomCallback(Callback):
    other_cb_functions = """
        def on_train_begin(self, logs=None):
        def on_train_end(self, logs=None):
        def on_epoch_begin(self, epoch, logs=None):
        def on_test_begin(self, logs=None):
        def on_test_end(self, logs=None):
        def on_predict_begin(self, logs=None):
        def on_predict_end(self, logs=None):
        def on_train_batch_begin(self, batch, logs=None):
        def on_train_batch_end(self, batch, logs=None):
        def on_test_batch_begin(self, batch, logs=None):
        def on_test_batch_end(self, batch, logs=None):
        def on_predict_batch_begin(self, batch, logs=None):
        def on_predict_batch_end(self, batch, logs=None):
    """

    def __init__(self, is_verbose: bool = False, is_stateful: bool = False):
        super().__init__()
        self.verbose = is_verbose
        self.is_stateful = is_stateful

    def on_epoch_end(self, epoch, logs=None):
        if self.is_stateful:
            verbose('Resetting LSTM states...', self.verbose)
            self.model.reset_states()


def parseKerasHistory(history: History) -> dict:
    new_hist = {}
    for key in list(history.history.keys()):
        new_hist[key] = numpyToListRecursive(history.history[key])
    return new_hist


def estimateLstmLayerOutputSize(layer_input_size: int, network_output_size: int, amount_train_samples=0, a: int = 2,
                                use_alternative_formula: bool = False) -> int:
    if not use_alternative_formula:
        return int(math.ceil(amount_train_samples / (a * (layer_input_size + network_output_size))))
    else:
        return int(math.ceil(2 / 3 * (layer_input_size + network_output_size)))


class Prophet(object):
    pass  # Just to hint its return type in the true class


class Prophet(object):
    __create_key = object()
    _new_model_counter = 0

    PARALLELISM = 1  # 1 means no parallelism, 0 means all cores

    def __init__(self, basename: str, model: Sequential(), callbacks: list[Callback],
                 configs: Hyperparameters, do_verbose: Optional[bool] = None, path_subdir: str = "",
                 _create_key: Optional[object] = None):
        assert (_create_key == Prophet.__create_key), \
            "The Prophet must be summoned using Prophet.build(...)! This constructor is private!"
        self.basename = basename
        self.model = model
        self.callbacks = callbacks
        self.configs = configs
        self.history = {}
        self.metrics = {}
        self.scaler_path = None
        if do_verbose is None:
            do_verbose = getVerbose()
        self.verbose = do_verbose
        self.path_subdir = path_subdir
        if path_subdir != "":
            Prophet.crateDirs(path_subdir)

    def drawAndSaveArchitecture(self, show_types: bool = False, ignore_error: bool = False):
        try:
            plot_model(self.model, to_file=self.getArchitectureFilepath(), show_shapes=True, show_layer_names=True,
                       rankdir="TB", show_layer_activations=True, expand_nested=False, dpi=FIGURE_DPI,
                       show_dtype=show_types)
        except ImportError:
            error(f'Could not draw {self.basename} architecture, probably `graphviz` is not installed on OS, '
                  f'check: https://graphviz.gitlab.io/download/!')
        except Exception as e:
            if not ignore_error:
                raise e

    def save(self, ignore_error: bool = False, do_log: bool = False) -> str:
        info(f'Saving {self.basename} prophet...', do_log)
        prophet = {
            'basename': self.basename,
            'saved_at': getNowStr(),
            'paths': {
                'prophet': self.getProphetFilepath(),
                'hyperparameters': self.configs.getConfigFilepath(),
                'model': self.getModelFilepath(),
                'checkpoint': self.getCheckpointFilepath(),
                'metrics': self.getMetricsFilepath(),
                'arch': self.getArchitectureFilepath(),
                'history': self.getHistoryFilepath(),
                'scaler': self.scaler_path,
                'dataset': getTransformedTickerFilepath(
                    getBasename(self.scaler_path)) if self.scaler_path is not None else None,
            }
        }
        try:
            verbose('Saving model...', do_log)
            runWithExpRetry(f'SaveModel', self.model.save, [prophet["paths"]["model"]], {}, 3)
            verbose(f'Saved model at `{prophet["paths"]["model"]}`!', do_log)
        except Exception as e:
            if not ignore_error:
                raise e

        try:
            verbose('Saving hyperparameters...', do_log)
            self.configs.saveJson()  # already has exceptionExpRetry
            verbose(f'Saved hyperparameters at `{prophet["paths"]["hyperparameters"]}`!', do_log)
        except Exception as e:
            if not ignore_error:
                raise e

        try:
            verbose('Saving training history...', do_log)
            runWithExpRetry(f'SaveModelHistory', saveJson, [self.history, prophet["paths"]["history"]], {}, 3)
            verbose(f'Saved training history at `{prophet["paths"]["history"]}`!', do_log)
        except Exception as e:
            if not ignore_error:
                raise e

        try:
            verbose('Saving eval metrics...', do_log)
            runWithExpRetry(f'SaveMetrics', saveJson, [self.metrics, prophet["paths"]["metrics"]], {}, 3)
            verbose(f'Saved eval metrics at `{prophet["paths"]["metrics"]}`!', do_log)
        except Exception as e:
            if not ignore_error:
                raise e
        try:
            runWithExpRetry(f'SaveProphet', saveJson, [prophet, prophet["paths"]["prophet"]], dict(sort_keys=False),
                            3)
        except Exception as e:
            if not ignore_error:
                raise e

        info(f'Saved {self.basename} prophet!', do_log)
        return prophet["paths"]["prophet"]

    def restoreCheckpoint(self, force: bool = False, delete_cp_after: bool = True, do_log: bool = True) -> None:
        cp_filepath = self.getCheckpointFilepath()
        if pathExists(cp_filepath) and (force or not pathExists(self.getOverwrittenByCpModelFilepath())):
            info(f'Restoring {getBasename(cp_filepath)} checkpoint...', do_log)
            loaded_model = load_model(cp_filepath, custom_objects=getAllCustomMetrics())
            if self.model is None:
                self.model = loaded_model
            else:
                self.model.set_weights(loaded_model.get_weights())
            if pathExists(self.getModelFilepath()):
                moveFile(self.getModelFilepath(), self.getOverwrittenByCpModelFilepath())
            copyFile(cp_filepath, self.getModelFilepath())
            if pathExists(self.getMetricsFilepath()):
                moveFile(self.getMetricsFilepath(), self.getOverwrittenByCpMetricsFilepath())
            if delete_cp_after:
                deleteFile(cp_filepath)
            info(f'Restored {getBasename(cp_filepath)} checkpoint!', do_log)

    def train(self, processed_data: ProcessedDataset, do_plot: bool = True, do_log: bool = True) -> dict:
        batch_size = None

        if processed_data.hasTrain():
            train_x = processed_data.features_train
            train_y = processed_data.labels_train
        else:
            raise AttributeError('Invalid splitted_data')
        if processed_data.hasTrain():
            validation_data = [processed_data.features_val, processed_data.labels_val]
        else:
            validation_data = None

        if self.configs.network.batch_size > 0:
            batch_size = self.configs.network.batch_size
            if self.configs.network.batch_size > 1:
                new_size_train = int(train_x.shape[0] / batch_size) * batch_size
                train_x = train_x[:new_size_train]
                train_y = train_y[:new_size_train]
                if validation_data is not None:
                    new_size_val = int(validation_data[0].shape[0] / batch_size) * batch_size
                    validation_data[0] = validation_data[0][:new_size_val]
                    validation_data[1] = validation_data[1][:new_size_val]
        info(f'Training prophet model `{self.basename}`...', do_log)
        if Prophet.PARALLELISM == 0:
            workers = getCpuCount()
        elif Prophet.PARALLELISM <= 0:
            workers = max(getCpuCount() + Prophet.PARALLELISM, 1)
        else:
            workers = Prophet.PARALLELISM
        history = self.model.fit(train_x, train_y, epochs=self.configs.network.max_epochs,
                                 validation_data=validation_data, batch_size=batch_size, callbacks=self.callbacks,
                                 shuffle=self.configs.network.shuffle, verbose=2 if self.verbose else 0,
                                 workers=workers, use_multiprocessing=Prophet.PARALLELISM != 1)
        self.restoreCheckpoint(do_log=do_log)
        info(f'Trained prophet model `{self.basename}`!', do_log)
        history = parseKerasHistory(history)
        self.history = history
        self.scaler_path = processed_data.scaler_path

        if do_plot:
            self._plotTrainHistory()
        return self.history

    def predict(self, processed_data: ProcessedDataset, scale_back: bool = True,
                do_log: bool = True) -> Union[dict, list]:
        info(f'Predicting model `{self.basename}` for `{processed_data.ticker}` ticker...', do_log)
        model = self.batchSizeWorkaround(do_log=do_log)  # needed to avoid cropping test data

        data_to_predict = processed_data.getFeaturesDict()
        if len(data_to_predict) == 0:
            raise AttributeError('No data found to be evaluated.')

        results = {}
        for d_type, data in data_to_predict.items():
            predictions = self.model.predict(data, batch_size=self.configs.network.batch_size,
                                             verbose=1 if self.verbose else 0)
            results[d_type] = decodeWindowedPredictions(predictions, self.configs.network.backward_samples,
                                                        self.configs.network.forward_samples)
        self.model = model  # restoring model
        info(f'Predicted model `{self.basename}` for `{processed_data.ticker}` ticker!', do_log)

        if scale_back:
            self.scaleBack(results)

        if len(results) == 1:
            return list(results.values())[0]
        else:
            return results

    def evaluate(self, processed_data: ProcessedDataset, do_log: bool = True) -> Union[dict, list]:
        info(f'Evaluating model `{self.basename}` for `{processed_data.ticker}` ticker...', do_log)
        model = self.batchSizeWorkaround(do_log=do_log)  # needed to avoid cropping test data

        data_to_evaluate = processed_data.getFeaturesAndLabelsDict()
        if len(data_to_evaluate) == 0:
            raise AttributeError('No data found to be evaluated.')

        metrics_names = ['loss'] + getKerasRegressionMetrics(get_names_only=True)
        results = {}
        for d_type, data in data_to_evaluate.items():
            keras_metrics = self.model.evaluate(data[0], data[1], batch_size=self.configs.network.batch_size,
                                                verbose=1 if self.verbose else 0)
            keras_metrics = {metrics_names[i]: val for i, val in enumerate(keras_metrics)}
            results[d_type] = keras_metrics
        self.model = model  # restoring model
        info(f'Evaluated model `{self.basename}` for `{processed_data.ticker}` ticker!', do_log)

        if len(results) == 1:
            return list(results.values())[0]
        else:
            return results

    def prophesize(self, processed_data: ProcessedDataset, do_plot: bool = True, do_log: bool = True) -> dict:
        info(f'Prophesizing model `{self.basename}` for `{processed_data.ticker}` ticker...', do_log)
        predictions = self.predict(processed_data, do_log=do_log)
        labels = self.labels(processed_data)
        tf_metrics = self.evaluate(processed_data, do_log=do_log)
        manual_metrics = {}
        all_metrics_and_values = computeMetricsAndGetValues(predictions, labels, processed_data.getPreviousLabelsDict(),
                                                            processed_data.index_tracker, processed_data.dates_mapper,
                                                            threshold=self.configs.pred_threshold)
        for d_type, all_data in all_metrics_and_values.items():
            values = []
            future_dates = None
            future_values = []
            pr_graph = []
            roc_graph = []
            cm_graph = []
            manual_metrics[f'{d_type}'] = {}
            for agg_method, data in all_data['predictions'].items():
                manual_metrics[f'{d_type}'][AggregationMethod.strNoSpace(agg_method)] = data['metrics']
                values.append((agg_method, data['vals']))
                if data['metrics']['binary']['graphs']['pr'] is not None:
                    precisions = data['metrics']['binary']['graphs']['pr']['precisions']
                    recalls = data['metrics']['binary']['graphs']['pr']['recalls']
                    if precisions is not None and recalls is not None:
                        pr_graph.append([agg_method, [recalls, precisions]])
                if data['metrics']['binary']['graphs']['roc'] is not None:
                    false_pos = data['metrics']['binary']['graphs']['roc']['false_pos']
                    true_pos = data['metrics']['binary']['graphs']['roc']['true_pos']
                    if false_pos is not None and true_pos is not None:
                        roc_graph.append([agg_method, [false_pos, true_pos]])
                if data['metrics']['binary']['graphs']['cm'] is not None:
                    cm_graph.append([agg_method, data['metrics']['binary']['graphs']['cm']])
                future = data.get('future', None)
                if future is not None:
                    if future_dates is None:
                        future_dates = future['dates']
                    future_values.append((agg_method, future['vals']))
            if do_plot:
                if len(values) > 0:
                    self._plotDatasetPredictions(values, all_data, processed_data.ticker, d_type)
                if len(future_values) > 0:
                    self._plotFuturePredictions(future_dates, future_values, all_data, processed_data.ticker)
                if len(pr_graph) > 0:
                    self._plotBinaryCurves(pr_graph, processed_data.ticker, d_type, 'pr', 'Recall (Coverage)',
                                           'Precision (Efficiency)')
                if len(roc_graph) > 0:
                    self._plotBinaryCurves(pr_graph, processed_data.ticker, d_type, 'roc', 'False Positive Rate',
                                           'True Positive Rate')
                if len(cm_graph) > 0:
                    self._plotConfusionMatrix(cm_graph, d_type, processed_data.ticker)
        self.metrics = {
            'model_metrics': {str(k): v for k, v in tf_metrics.items()},
            'manual_metrics': manual_metrics
        }
        verbose('Saving metrics...', do_log)
        runWithExpRetry(f'SaveProphesizedMetrics', saveJson, [self.metrics, self.getMetricsFilepath()], {}, 3)
        verbose(f'Saved metrics at `{self.getMetricsFilepath()}`!', do_log)
        info(f'Prophesied model `{self.basename}` for `{processed_data.ticker}` ticker!', do_log)
        return self.metrics

    def continuousPrediction(self, data: Union[np.ndarray, ProcessedDataset], days: int, do_log: bool = True,
                             do_plot: bool = True) -> dict:
        # TODO, we need to predict also the Volume, Open, High, Low to be able to generated the next fields and enrich
        #  we need to have model groups
        raise NotImplemented()

    def _plotTrainHistory(self):
        plot_data = []
        to_plot = False
        if 'loss' in self.history:
            plot_data.append(
                ('line', [list(range(len(self.history['loss']))), self.history['loss']], {'label': f'Loss'}))
            to_plot = True
        if 'val_loss' in self.history:
            plot_data.append(('line', [list(range(len(self.history['val_loss']))), self.history['val_loss']],
                              {'label': f'Validation loss'}))
            to_plot = True
        if to_plot:
            plot(plot_data, title='Training loss per epoch', x_label='epoch', y_label='loss', legend=True,
                 subdir=pathJoin(self.path_subdir, 'training_history'), add_rid_subdir=False,
                 file_prefix=False, file_postfix=False, file_label=self.basename)

    def _plotDatasetPredictions(self, values: list, all_data: dict, ticker: str, d_type: DatasetSplit) -> None:
        max_plots_per_graph = 5
        true_value_color = 'b'
        plot_prefix = removeFileExtension(getBasename(self.scaler_path))
        plot_chunks = listToChunks(values, chunk_sz=max_plots_per_graph, filter_empty=True)
        for c, chunk in enumerate(plot_chunks):
            plot_data = [
                ('line', [all_data['dates'], all_data['label_vals']],
                 {'label': f'True price', 'zorder': 2, 'color': true_value_color}),
            ]
            for i, to_plot in enumerate(chunk):
                plot_data.append(
                    ('line', [all_data['dates'], to_plot[1]],
                     {'label': f'{to_plot[0]}', 'zorder': 1,
                      'color': getPlotColorFromIndex(i, colours_to_avoid=true_value_color)}),
                )
            title = f'Stock values for `{ticker}` - {d_type}'
            file_label = f'stock_val-{d_type}'
            if len(plot_chunks) > 1:
                title += f' - {c + 1} of {len(plot_chunks)}'
                file_label += f'-{c + 1}of{len(plot_chunks)}'
            plot(plot_data, title=title, y_label='Close price (USD)', x_label='Date', legend=True,
                 legend_outside=.3, resize=True, subdir=pathJoin(self.path_subdir, 'model_output'),
                 add_rid_subdir=False, file_prefix=plot_prefix, file_postfix=False, file_label=file_label)

    def _plotFuturePredictions(self, future_dates: list, future_values: list, all_data: dict,
                               ticker: str, previous_data_points: int = 5) -> None:
        max_plots_per_graph = 5
        plot_chunks = listToChunks(future_values, chunk_sz=max_plots_per_graph, filter_empty=True)
        true_value_color = 'b'
        plot_prefix = removeFileExtension(getBasename(self.scaler_path))
        for c, chunk in enumerate(plot_chunks):
            plot_data = []
            if previous_data_points > 0:
                plot_data.append(
                    ('line', [all_data['dates'][-previous_data_points:],
                              all_data['label_vals'][-previous_data_points:]],
                     {'label': f'True price', 'zorder': 1, 'style': '-o', 'color': true_value_color})
                )
            for i, to_plot in enumerate(chunk):
                plot_data.append(
                    ('line', [future_dates, to_plot[1]],
                     {'label': f'pred {to_plot[0]}', 'zorder': 2, 'style': '-o',
                      'color': getPlotColorFromIndex(i, colours_to_avoid=true_value_color)}),
                )
            title = f'Predicted future prices for `{ticker}`'
            file_label = f'future_prediction'
            if len(plot_chunks) > 1:
                title += f' - {c + 1} of {len(plot_chunks)}'
                file_label += f'-{c + 1}of{len(plot_chunks)}'
            x_ticks_args = {'rotation': 30, 'ha': 'right'}
            if previous_data_points > 0:
                x_ticks = (all_data['dates'][-previous_data_points:] + future_dates, x_ticks_args)
            else:
                x_ticks = (future_dates, x_ticks_args)
            plot(plot_data, title=title, y_label='Close price (USD)', x_label='Date', legend=True,
                 legend_outside=True, resize=True, x_ticks=x_ticks, subdir=pathJoin(self.path_subdir, 'model_output'),
                 add_rid_subdir=False, file_prefix=plot_prefix, file_postfix=False, file_label=file_label)

    def _plotBinaryCurves(self, graphs: list, ticker: str, d_type: DatasetSplit,
                          curve_name: str, x_label: str, y_label: str) -> None:
        max_plots_per_graph = 13
        plot_prefix = removeFileExtension(getBasename(self.scaler_path))
        plot_chunks = listToChunks(graphs, chunk_sz=max_plots_per_graph, filter_empty=True)
        for c, chunk in enumerate(plot_chunks):
            plot_data = []
            for i, to_plot in enumerate(chunk):
                plot_data.append(
                    ('line', to_plot[1], {'label': f'{to_plot[0]}', 'color': getPlotColorFromIndex(i)}))
            title = f'{curve_name.upper()} Curve for `{ticker}` - {d_type}'
            file_label = f'{curve_name.lower()}_curve_{d_type}'
            ticks = ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], {})
            if len(plot_chunks) > 1:
                title += f' - {c + 1} of {len(plot_chunks)}'
                file_label += f'-{c + 1}of{len(plot_chunks)}'
            plot(plot_data, title=title, x_label=x_label, y_label=y_label,
                 x_ticks=ticks, y_ticks=ticks, legend=True, legend_outside=True, resize=True,
                 subdir=pathJoin(self.path_subdir, 'binary_metrics'), add_rid_subdir=False, file_prefix=plot_prefix,
                 file_postfix=False,
                 file_label=file_label)

    def _plotConfusionMatrix(self, cm_graphs: list, d_type: DatasetSplit, ticker: str) -> None:
        plot_prefix = removeFileExtension(getBasename(self.scaler_path))
        for i, (agg_method, cm) in enumerate(cm_graphs):
            title = f'Confusion Matrix Curve for `{ticker}` - {d_type}'
            subtitle = f'{agg_method}'
            line_kwargs = dict(color='black', lw=1)
            text_kwargs = dict(horizontalalignment='center', verticalalignment='center',
                               bbox=dict(fill=True, facecolor='white', edgecolor='white', linewidth=2,
                                         boxstyle='round'))
            plot_data = [
                ('im', [cm],
                 {'interpolation': 'nearest', 'cmap': getColorGradientsFromIndex(2, True),  # 'auto_range': True,
                  'vmin': 0, 'vmax': int(sum([sum(x) for x in cm]) / 2)  # 'auto_range_factors': (1, .95)
                  }),
                ('colorbar', None, {}),
                ('suptitle', [title], {}),
                ('axhline', [0.5], line_kwargs),
                ('axvline', [0.5], line_kwargs),
                ('text', [0, 0, f'{cm[0][0]}'], text_kwargs),
                ('text', [1, 0, f'{cm[0][1]}'], text_kwargs),
                ('text', [0, 1, f'{cm[1][0]}'], text_kwargs),
                ('text', [1, 1, f'{cm[1][1]}'], text_kwargs)]
            file_label = f'cm_{d_type}_{AggregationMethod.strNoSpace(agg_method)}_'
            x_ticks = (np.arange(0, 2), ['Positive', 'Negative'], {})
            y_ticks = (np.arange(0, 2), ['Positive', 'Negative'], {'rotation': 90})
            x_label = 'Predicted'
            y_label = 'True'
            plot(plot_data, title=subtitle, x_label=x_label, y_label=y_label,
                 x_ticks=x_ticks, y_ticks=y_ticks, legend=False,
                 subdir=pathJoin(self.path_subdir, 'binary_metrics/confusion_matrix'), add_rid_subdir=False,
                 file_prefix=plot_prefix, file_postfix=False, file_label=file_label)

    def labels(self, processed_data: ProcessedDataset, scale_back: bool = True) -> Union[dict, list]:
        labels = {}
        if processed_data.hasTrain():
            labels[DatasetSplit.TRAIN] = processed_data.labels_train
        if processed_data.hasVal():
            labels[DatasetSplit.VALIDATION] = processed_data.labels_val
        if processed_data.hashEval():
            labels[DatasetSplit.TEST] = processed_data.labels_test
        if len(labels) == 0:
            raise AttributeError('No labels found.')

        results = {}
        for d_type, data in labels.items():
            results[d_type] = decodeWindowedLabels(data, self.configs.network.backward_samples)
        if scale_back:
            self.scaleBack(results)

        if len(results) == 1:
            return list(results.values())[0]
        else:
            return results

    def isTrained(self) -> bool:
        return len(self.history) > 0 and self.scaler_path is not None

    def batchSizeWorkaround(self, do_log: bool = True) -> Sequential:
        info(f'Casting `{self.basename}` to stateless model with batch_size = 1...', do_log)
        new_prophet = Prophet.build(self.configs, self.basename, workaround_for_eval=True, do_log=False)
        new_prophet.model.set_weights(self.model.get_weights())
        old_model = self.model
        self.model = new_prophet.model
        info(f'Casted `{self.basename}` to stateless model with batch_size = 1!', do_log)
        return old_model

    def scaleBack(self, dict_of_prices: dict) -> None:
        if self.scaler_path is not None and pathExists(self.scaler_path):
            scaler = runWithExpRetry(f'LoadScaler', loadObj, [self.scaler_path], {}, 3)[2]
            if scaler is not None:
                for k in list(dict_of_prices.keys()):
                    for i in range(len(dict_of_prices[k])):
                        if dict_of_prices[k][i] is not None:
                            numeric_item = type(dict_of_prices[k][i]) in getNumericTypes()
                            to_scale = np.array(dict_of_prices[k][i])
                            to_scale = to_scale.reshape(tuple(list(to_scale.shape) + [-1]))
                            if len(to_scale.shape) == 1:
                                to_scale = to_scale.reshape(1, -1)
                            axis = None if numeric_item else 1
                            scaled = np.squeeze(scaler.inverse_transform(to_scale), axis=axis).tolist()
                            dict_of_prices[k][i] = scaled
        else:
            error(f'Could not scale back prices, could not find scaler at `{self.scaler_path}`')

    def getModelFilepath(self) -> str:
        return Prophet._getModelFilepath(self.basename, self.path_subdir)

    def getCheckpointFilepath(self) -> str:
        return Prophet._getCheckpointFilepath(self.basename, self.path_subdir)

    def getOverwrittenByCpModelFilepath(self) -> str:
        return Prophet._getOverwrittenByCpModelFilepath(self.basename, self.path_subdir)

    def getArchitectureFilepath(self) -> str:
        return Prophet._getArchitectureFilepath(self.basename, self.path_subdir)

    def getHistoryFilepath(self) -> str:
        return Prophet._getHistoryFilepath(self.basename, self.path_subdir)

    def getMetricsFilepath(self) -> str:
        return Prophet._getMetricsFilepath(self.basename, self.path_subdir)

    def getOverwrittenByCpMetricsFilepath(self) -> str:
        return Prophet._getOverwrittenByCpMetricsFilepath(self.basename, self.path_subdir)

    def getProphetFilepath(self) -> str:
        return Prophet.getProphetFilepathFromBasename(self.basename, self.path_subdir)

    def enhancedLoss(self, loss_name):
        # TODO ADAPT THIS TO OUR CONTEXT
        USE_ALL_INSTEAD_OF_ANY = True
        LOSS_NEGATIVE_LABEL_WEIGHT = 0.3
        LOSS_POSITIVE_LABEL_WEIGHT = 1.0

        def loss(y_true, y_pred):
            loss_fn = tf.keras.losses.get(loss_name)
            loss_v = loss_fn(y_true, y_pred)
            loss_negative_label = loss_v * LOSS_NEGATIVE_LABEL_WEIGHT
            loss_positive_label = loss_v * LOSS_POSITIVE_LABEL_WEIGHT
            cond = tf.keras.backend.greater_equal(y_true, 1)
            if USE_ALL_INSTEAD_OF_ANY:
                cond = tf.keras.backend.all(cond, axis=1)
            else:
                cond = tf.keras.backend.any(cond, axis=1)
            # returns loss_positive_label when all y_true>=1 (or one of them for any)
            loss_v = tf.keras.backend.switch(cond, loss_positive_label, loss_negative_label)
            return loss_v

        return loss

    @staticmethod
    def genProphetBasename(configs: Hyperparameters, basename: Optional[str] = None, include_rid: bool = False,
                           include_counter: bool = False, include_net_id: bool = True) -> str:
        if basename is None or basename.strip() == '':
            if configs.name is not None:
                basename = f'lstm_model{f"-{getRunIdStr()}-" if include_rid else ""}{configs.name}'
            else:
                basename = f'lstm_model{f"-{getRunIdStr()}-" if include_rid else ""}' \
                           f'-{str(Prophet._new_model_counter) if include_counter else ""}'
                if include_counter:
                    Prophet._new_model_counter += 1
        elif configs.name is not None and basename != configs.name:
            basename = f'{basename}-{configs.name}'
        if include_net_id:
            basename += f'-{configs.network_uuid}'
        return basename

    @staticmethod
    def crateDirs(path_subdir: str = ""):
        createFolder(PROPHET_DIR)
        createFolder(pathJoin(PROPHET_DIR, path_subdir))
        createFolder(MODELS_DIR)
        createFolder(pathJoin(MODELS_DIR, path_subdir))
        createFolder(Prophet._getCheckpointFilepath('', path_subdir))
        createFolder(Prophet._getArchitectureFilepath('', path_subdir))
        createFolder(Prophet._getHistoryFilepath('', path_subdir))
        createFolder(Prophet._getMetricsFilepath('', path_subdir))

    @staticmethod
    def _getModelFilepath(basename: str, path_subdir: str = "") -> str:
        filename_no_ext = removeFileExtension(basename)
        filename_changed = ''
        if not filename_no_ext.strip().endswith('.') and filename_no_ext.strip() != '':
            filename_changed = f'{filename_no_ext}.h5'
        return pathJoin(MODELS_DIR, path_subdir, filename_changed)

    @staticmethod
    def _getCheckpointFilepath(basename: str, path_subdir: str = "") -> str:
        filename_no_ext = removeFileExtension(basename)
        filename_changed = ''
        if not filename_no_ext.strip().endswith('.') and filename_no_ext.strip() != '':
            # '_{epoch:06d}_{val_loss}' # TODO for this i have to search for the best loss when loading
            keras_placeholders = ''
            filename_changed = f'{filename_no_ext}' + keras_placeholders + '_cp.h5'
        return pathJoin(MODELS_DIR, path_subdir, CHECKPOINT_SUBDIR, filename_changed)

    @staticmethod
    def _getOverwrittenByCpModelFilepath(basename: str, path_subdir: str = "") -> str:
        filename_no_ext = removeFileExtension(basename)
        filename_changed = ''
        if not filename_no_ext.strip().endswith('.') and filename_no_ext.strip() != '':
            filename_changed = f'{filename_no_ext}_last_patience_model.h5'
        return pathJoin(MODELS_DIR, path_subdir, CHECKPOINT_SUBDIR, filename_changed)

    @staticmethod
    def _getArchitectureFilepath(basename: str, path_subdir: str = "") -> str:
        filename_no_ext = removeFileExtension(basename)
        filename_changed = ''
        if not filename_no_ext.strip().endswith('.') and filename_no_ext.strip() != '':
            filename_changed = f'{filename_no_ext}_arch.png'
        return pathJoin(MODELS_DIR, path_subdir, ARCHITECTURE_SUBDIR, filename_changed)

    @staticmethod
    def _getHistoryFilepath(basename: str, path_subdir: str = "") -> str:
        filename_no_ext = removeFileExtension(basename)
        filename_changed = ''
        if not filename_no_ext.strip().endswith('.') and filename_no_ext.strip() != '':
            filename_changed = f'{filename_no_ext}_train-history.json'
        return pathJoin(MODELS_DIR, path_subdir, HISTORY_SUBDIR, filename_changed)

    @staticmethod
    def _getMetricsFilepath(basename: str, path_subdir: str = "") -> str:
        filename_no_ext = removeFileExtension(basename)
        filename_changed = ''
        if not filename_no_ext.strip().endswith('.') and filename_no_ext.strip() != '':
            filename_changed = f'{filename_no_ext}_metrics.json'
        return pathJoin(MODELS_DIR, path_subdir, METRICS_SUBDIR, filename_changed)

    @staticmethod
    def _getOverwrittenByCpMetricsFilepath(basename: str, path_subdir: str = "") -> str:
        filename_no_ext = removeFileExtension(basename)
        filename_changed = ''
        if not filename_no_ext.strip().endswith('.') and filename_no_ext.strip() != '':
            filename_changed = f'{filename_no_ext}_last_patience_metrics.json'
        return pathJoin(MODELS_DIR, path_subdir, METRICS_SUBDIR, filename_changed)

    @staticmethod
    def getProphetFilepathFromBasename(basename: str, path_subdir: str = "") -> str:
        filename_no_ext = removeFileExtension(basename)
        filename_changed = ''
        if not filename_no_ext.strip().endswith('.') and filename_no_ext.strip() != '':
            filename_changed = f'{filename_no_ext}.json'
        return pathJoin(PROPHET_DIR, path_subdir, filename_changed)

    @staticmethod
    def destroy(model: Optional[object]) -> None:
        if model is not None:
            try:
                del model
            except:
                pass
        keras.backend.clear_session()

    @staticmethod
    def build(configs: Hyperparameters, basename: Optional[str] = None, workaround_for_eval: bool = False,
              do_log: bool = True, counter_on_basename: bool = True, do_verbose: bool = True,
              path_subdir: str = '', ignore_save_error: bool = False) -> Prophet:
        basename = Prophet.genProphetBasename(configs, basename, include_counter=counter_on_basename)
        info(f'Building {basename} prophet...', do_log)
        n_tickers = 1  # TODO support for training stocks together
        binary_classifier = False  # TODO support binary classification
        time_major = False
        model = Sequential(name=basename)
        input_features_size = configs.dataset.n_features
        batch_size_from_configs = configs.network.batch_size if not workaround_for_eval else 1
        for l in range(configs.network.n_hidden_lstm_layers):
            is_stateful = configs.network.stateful and not workaround_for_eval
            input_shape = (configs.network.layer_sizes[l], n_tickers * input_features_size)
            if l + 1 < configs.network.n_hidden_lstm_layers:
                return_sequences = True
            else:
                return_sequences = not configs.network.dense_instead_lstm_on_out
            batch_input_shape = None
            if l == 0 and not workaround_for_eval:
                batch_size = batch_size_from_configs
                if batch_size == 0:
                    batch_size = None
                batch_input_shape = tuple([batch_size]) + input_shape
            activation_function = configs.network.activation_funcs[l]
            rec_activation_function = configs.network.rec_activation_funcs[l]
            advanced_activation = None
            if activation_function == ActivationFunc.LEAKY_RELU:
                advanced_activation = activation_function
                activation_function = ActivationFunc.LINEAR
            if rec_activation_function == ActivationFunc.LEAKY_RELU:
                raise ValueError(f'Recurrent activation cannot be {rec_activation_function}')
            lstm_kwargs = dict(stateful=is_stateful, return_sequences=return_sequences,
                               use_bias=configs.network.use_bias[l],
                               activation=activation_function.toKerasName(),
                               recurrent_activation=rec_activation_function.toKerasName(),
                               unit_forget_bias=configs.network.unit_forget_bias[l],
                               recurrent_dropout=configs.network.rec_dropout[l],
                               go_backwards=configs.network.go_backwards[l], time_major=time_major,
                               name=f'lstm_{f"h_{l}" if l > 0 else "i"}',
                               kernel_regularizer=L1L2(l1=configs.network.kernel_l1_regularizer[l],
                                                       l2=configs.network.kernel_l2_regularizer[l]),
                               recurrent_regularizer=L1L2(l1=configs.network.recurrent_l1_regularizer[l],
                                                          l2=configs.network.recurrent_l2_regularizer[l]),
                               bias_regularizer=L1L2(l1=configs.network.bias_l1_regularizer[l],
                                                     l2=configs.network.bias_l2_regularizer[l]),
                               activity_regularizer=L1L2(l1=configs.network.activity_l1_regularizer[l],
                                                         l2=configs.network.activity_l2_regularizer[l]))
            if batch_input_shape is not None:
                lstm_kwargs['batch_input_shape'] = batch_input_shape
            else:
                lstm_kwargs['input_shape'] = input_shape
            model.add(LSTM(configs.network.layer_sizes[l + 1], **lstm_kwargs))
            if advanced_activation is not None:
                if advanced_activation == ActivationFunc.LEAKY_RELU:
                    model.add(LeakyReLU(alpha=configs.network.leaky_relu_alpha, name=f'lrelu_{l}'))
                else:
                    raise ValueError(f'Unsupported advanced activation: {advanced_activation}')
            if configs.network.dropout[l] > 0:
                model.add(Dropout(configs.network.dropout[l], name=f'drop_{l}'))
        if binary_classifier:
            output_activation = configs.network.binary_out_activation_func
        else:
            output_activation = configs.network.regression_out_activation_func  # activation=None = 'linear'
        if configs.network.n_hidden_lstm_layers > 0:
            if configs.network.dense_instead_lstm_on_out:
                model.add(Dense(configs.network.forward_samples * n_tickers,
                                activation=output_activation.toKerasName(), name='dense_out',
                                kernel_regularizer=L1L2(
                                    l1=configs.network.kernel_l1_regularizer[configs.network.n_hidden_lstm_layers],
                                    l2=configs.network.kernel_l2_regularizer[configs.network.n_hidden_lstm_layers]),
                                bias_regularizer=L1L2(
                                    l1=configs.network.bias_l1_regularizer[configs.network.n_hidden_lstm_layers],
                                    l2=configs.network.bias_l2_regularizer[configs.network.n_hidden_lstm_layers]),
                                activity_regularizer=L1L2(
                                    l1=configs.network.activity_l1_regularizer[configs.network.n_hidden_lstm_layers],
                                    l2=configs.network.activity_l2_regularizer[configs.network.n_hidden_lstm_layers])))
            else:
                model.add(LSTM(configs.network.forward_samples * n_tickers,
                               activation=output_activation.toKerasName(), time_major=time_major, name='lstm_out'))
        else:  # No dense layer for n_hidden_lstm_layers == 0
            input_shape = (
                configs.network.backward_samples, n_tickers * input_features_size)
            batch_size = batch_size_from_configs
            if batch_size <= 1:
                batch_size = None
            batch_input_shape = tuple([batch_size]) + input_shape
            model.add(LSTM(configs.network.forward_samples * n_tickers,
                           batch_input_shape=batch_input_shape, activation=output_activation.toKerasName(),
                           time_major=time_major, name='lstm_out',
                           kernel_regularizer=L1L2(
                               l1=configs.network.kernel_l1_regularizer[configs.network.n_hidden_lstm_layers],
                               l2=configs.network.kernel_l2_regularizer[configs.network.n_hidden_lstm_layers]),
                           recurrent_regularizer=L1L2(
                               l1=configs.network.recurrent_l1_regularizer[configs.network.n_hidden_lstm_layers],
                               l2=configs.network.recurrent_l2_regularizer[configs.network.n_hidden_lstm_layers]),
                           bias_regularizer=L1L2(
                               l1=configs.network.bias_l1_regularizer[configs.network.n_hidden_lstm_layers],
                               l2=configs.network.bias_l2_regularizer[configs.network.n_hidden_lstm_layers]),
                           activity_regularizer=L1L2(
                               l1=configs.network.activity_l1_regularizer[configs.network.n_hidden_lstm_layers],
                               l2=configs.network.activity_l2_regularizer[configs.network.n_hidden_lstm_layers])))

        if do_log and do_verbose:
            model_summary_lines = []
            model.summary(print_fn=lambda x: model_summary_lines.append(x))
            model_summary_str = '\n'.join(model_summary_lines) + '\n'
            clean(model_summary_str, is_verbose=True, do_log=do_log)

        clip_dict = {}
        if configs.network.clip_norm_instead_of_value:
            clip_dict['clipnorm'] = configs.network.clip_norm
        else:
            clip_dict['clipvalue'] = configs.network.clip_value
        if configs.network.optimizer == Optimizer.ADAM:
            opt = Adam(**clip_dict)
        elif configs.network.optimizer == Optimizer.SGD:
            opt = SGD(**clip_dict)
        elif configs.network.optimizer == Optimizer.RMSPROP:
            opt = RMSprop(**clip_dict)
        else:
            raise ValueError(f'Unknown optimizer {configs.network.optimizer}')
        model.compile(loss=configs.network.loss.toKerasName(), optimizer=opt,
                      metrics=getKerasRegressionMetrics())

        callbacks = Prophet._genCallbacks(configs, basename, path_subdir=path_subdir, do_verbose=do_verbose)
        prophet = Prophet(basename, model, callbacks, configs, do_verbose=do_verbose, path_subdir=path_subdir,
                          _create_key=Prophet.__create_key)
        info(f'Built {basename} prophet!', do_log)
        if not workaround_for_eval:
            prophet.drawAndSaveArchitecture(ignore_error=ignore_save_error)
        return prophet

    @staticmethod
    def load(prophet_filepath: str, do_log: bool = True, do_verbose: bool = True, path_subdir: str = "") -> Prophet:
        info(f'Loading prophet from `{prophet_filepath}`...', do_log)
        prophet_meta = runWithExpRetry(f'LoadProphet', loadJson, [prophet_filepath], {}, 3)
        basename = prophet_meta['basename']
        model = runWithExpRetry(f'LoadModel', load_model, [prophet_meta['paths']['model']],
                                dict(custom_objects=getAllCustomMetrics()), 3)
        configs = Hyperparameters.loadJson(prophet_meta['paths']['hyperparameters'])  # already has exceptionExpRetry
        callbacks = Prophet._genCallbacks(configs, basename, path_subdir=path_subdir, do_verbose=do_verbose)
        prophet = Prophet(basename, model, callbacks, configs, do_verbose=do_verbose, path_subdir=path_subdir,
                          _create_key=Prophet.__create_key)
        prophet.scaler_path = prophet_meta['paths']['scaler']
        if pathExists(prophet_meta['paths']['history']):
            prophet.history = runWithExpRetry(f'LoadHistory', loadJson, [prophet_meta['paths']['history']], {}, 3)
        if pathExists(prophet_meta['paths']['metrics']):
            prophet.metrics = runWithExpRetry(f'LoadMetrics', loadJson, [prophet_meta['paths']['metrics']], {}, 3)
        info(f'Loaded prophet!', do_log)
        return prophet

    @staticmethod
    def _genCallbacks(configs: Hyperparameters, basename: str, path_subdir: str = '', do_verbose: bool = True) -> list[
        Callback]:
        callbacks = []
        if configs.network.patience_epochs_stop > 0:
            early_stopping = EarlyStopping(monitor='val_loss', mode='auto',
                                           patience=configs.network.patience_epochs_stop,
                                           verbose=1 if do_verbose else 0)
            callbacks.append(early_stopping)
        if configs.network.patience_epochs_reduce > 0:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=configs.network.reduce_factor,
                                          patience=configs.network.patience_epochs_reduce,
                                          verbose=1 if do_verbose else 0)
            callbacks.append(reduce_lr)
        checkpoint_filedir = Prophet._getCheckpointFilepath('', path_subdir)
        createFolder(checkpoint_filedir)
        checkpoint_filepath = Prophet._getCheckpointFilepath(basename, path_subdir)
        checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1 if do_verbose else 0,
                                     save_best_only=True, mode='auto')
        callbacks.append(checkpoint)
        reset_states_after_epoch = CustomCallback(is_verbose=do_verbose, is_stateful=configs.network.stateful)
        callbacks.append(reset_states_after_epoch)
        return callbacks


tf.keras.backend.set_epsilon(EPSILON)
Prophet.crateDirs('')
