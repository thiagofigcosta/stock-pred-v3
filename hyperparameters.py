import re
import textwrap
from typing import Optional, Union

import numpy as np

from logger import fatal, warn
from prophet_enums import ActivationFunc, Optimizer, Loss
from search_space import SearchSpace
from utils_fs import getBasename, createFolder, pathJoin
from utils_misc import hashStr, getNumericTypes
from utils_persistance import loadJson, saveJson
from utils_random import randInt

HYPERPARAMETERS_DIR = 'hyperparameters'


class SubParameters(object):  # TODO: create class for each wrapper, I didn't since im lazy
    pass


class Hyperparameters(object):
    pass  # Just to hint its return type in the true class


class Hyperparameters(object):
    DEFAULT = None

    def __init__(self,
                 # uuid
                 uuid: Optional[str] = None,
                 misc_uuid: Optional[str] = None,
                 dataset_uuid: Optional[str] = None,
                 enricher_uuid: Optional[str] = None,
                 pca_uuid: Optional[str] = None,
                 network_uuid: Optional[str] = None,
                 # misc
                 name: Optional[str] = None,
                 dataset_filename: Optional[str] = None,
                 pred_threshold: Optional[float] = None,
                 # dataset
                 train_ratio: Optional[float] = None,
                 validation_ratio: Optional[float] = None,
                 n_features: Optional[int] = None,
                 normalize: Optional[bool] = None,
                 normalize_prediction_feat: Optional[bool] = None,
                 norm_range: Optional[Union[tuple[float, float], list[float, float]]] = None,
                 # enricher
                 price_ratios_on: Optional[bool] = None,
                 price_delta_on: Optional[bool] = None,
                 price_averages_on: Optional[bool] = None,
                 ta_lib_on: Optional[bool] = None,
                 fibonacci_seq_size: Optional[int] = None,
                 fast_average_window: Optional[int] = None,
                 slow_average_window: Optional[int] = None,
                 # pca
                 use_kernel_pca: Optional[bool] = None,
                 kernel_pca_type: Optional[str] = None,
                 pca_norm_range: Optional[Union[tuple[float, float], list[float, float]]] = None,
                 knee_sensibility: Optional[int] = None,
                 # network
                 backward_samples: Optional[int] = None,
                 forward_samples: Optional[int] = None,
                 max_epochs: Optional[int] = None,
                 n_hidden_lstm_layers: Optional[int] = None,
                 layer_sizes: Optional[Union[list[int], int]] = None,
                 use_bias: Optional[Union[list[bool], bool]] = None,
                 unit_forget_bias: Optional[Union[list[bool], bool]] = None,
                 activation_funcs: Optional[Union[list[int], list[ActivationFunc], int, ActivationFunc]] = None,
                 rec_activation_funcs: Optional[Union[list[int], list[ActivationFunc], int, ActivationFunc]] = None,
                 dropout: Optional[Union[list[float], float]] = None,
                 rec_dropout: Optional[Union[list[float], float]] = None,
                 kernel_l1_regularizer: Optional[Union[list[float], float]] = None,
                 bias_l1_regularizer: Optional[Union[list[float], float]] = None,
                 recurrent_l1_regularizer: Optional[Union[list[float], float]] = None,
                 activity_l1_regularizer: Optional[Union[list[float], float]] = None,
                 kernel_l2_regularizer: Optional[Union[list[float], float]] = None,
                 bias_l2_regularizer: Optional[Union[list[float], float]] = None,
                 recurrent_l2_regularizer: Optional[Union[list[float], float]] = None,
                 activity_l2_regularizer: Optional[Union[list[float], float]] = None,
                 stateful: Optional[bool] = None,
                 go_backwards: Optional[Union[list[bool], bool]] = None,
                 dense_instead_lstm_on_out: Optional[bool] = None,
                 batch_size: Optional[int] = None,
                 regression_out_activation_func: Optional[Union[ActivationFunc, int]] = None,
                 binary_out_activation_func: Optional[Union[ActivationFunc, int]] = None,
                 clip_norm_instead_of_value: Optional[bool] = None,
                 clip_norm: Optional[float] = None,
                 clip_value: Optional[float] = None,
                 optimizer: Optional[Union[Optimizer, int]] = None,
                 patience_epochs_stop: Optional[int] = None,
                 patience_epochs_reduce: Optional[int] = None,
                 reduce_factor: Optional[float] = None,
                 shuffle: Optional[bool] = None,
                 loss: Optional[Union[Loss, int]] = None,
                 leaky_relu_alpha: Optional[Union[Loss, float]] = None,
                 # parameters wrappers
                 dataset_parameters: Optional[SubParameters] = None,
                 enricher_parameters: Optional[SubParameters] = None,
                 pca_parameters: Optional[SubParameters] = None,
                 network_parameters: Optional[SubParameters] = None):
        # misc
        self.name = name
        self.dataset_filename = None
        if dataset_filename is not None:
            self.setDatasetFilename(dataset_filename)
        if pred_threshold is None:
            pred_threshold = .5
        self.pred_threshold = pred_threshold

        if dataset_parameters is None:
            dataset_parameters = SubParameters()
            if train_ratio is None:
                train_ratio = .7
            dataset_parameters.train_ratio = train_ratio

            if validation_ratio is None:
                validation_ratio = .2
            dataset_parameters.validation_ratio = validation_ratio

            if n_features is None:
                n_features = self.loadAmountOfFeatures()
            dataset_parameters.n_features = n_features

            if normalize is None:
                normalize = True
            dataset_parameters.normalize = normalize

            if normalize_prediction_feat is None:
                normalize_prediction_feat = False
            dataset_parameters.normalize_prediction_feat = normalize_prediction_feat

            if norm_range is None:
                norm_range = (0, 1)
            dataset_parameters.norm_range = norm_range
        self.dataset = dataset_parameters

        if enricher_parameters is None:
            enricher_parameters = SubParameters()

            if price_ratios_on is None:
                price_ratios_on = True
            enricher_parameters.price_ratios_on = price_ratios_on

            if price_delta_on is None:
                price_delta_on = True
            enricher_parameters.price_delta_on = price_delta_on

            if price_averages_on is None:
                price_averages_on = True
            enricher_parameters.price_averages_on = price_averages_on

            if ta_lib_on is None:
                ta_lib_on = True
            enricher_parameters.ta_lib_on = ta_lib_on

            if fibonacci_seq_size is None:
                fibonacci_seq_size = 10
            enricher_parameters.fibonacci_seq_size = fibonacci_seq_size

            if fast_average_window is None:
                fast_average_window = 13
            enricher_parameters.fast_average_window = fast_average_window

            if slow_average_window is None:
                slow_average_window = 21
            enricher_parameters.slow_average_window = slow_average_window
        self.enricher = enricher_parameters

        if pca_parameters is None:
            pca_parameters = SubParameters()
            if use_kernel_pca is None:
                use_kernel_pca = False
            pca_parameters.use_kernel_pca = use_kernel_pca

            if kernel_pca_type is None:
                kernel_pca_type = 'linear'
            pca_parameters.kernel_pca_type = kernel_pca_type

            if pca_norm_range is None:
                pca_norm_range = (-1, 1)
            pca_parameters.pca_norm_range = pca_norm_range

            if knee_sensibility is None:
                knee_sensibility = 13
            pca_parameters.knee_sensibility = knee_sensibility
        self.pca = pca_parameters

        if network_parameters is None:
            network_parameters = SubParameters()

            if backward_samples is None:
                backward_samples = 30
            network_parameters.backward_samples = backward_samples

            if forward_samples is None:
                forward_samples = 7
            network_parameters.forward_samples = forward_samples

            if max_epochs is None:
                max_epochs = 200
            network_parameters.max_epochs = max_epochs

            if n_hidden_lstm_layers is None:
                n_hidden_lstm_layers = 2
            network_parameters.n_hidden_lstm_layers = n_hidden_lstm_layers

            if layer_sizes is None:
                layer_sizes = [25, 15] if n_hidden_lstm_layers == 2 else [randInt(30, low=5) for _ in
                                                                          range(n_hidden_lstm_layers)]
            network_parameters.layer_sizes = layer_sizes

            if use_bias is None:
                use_bias = True
            network_parameters.use_bias = use_bias

            if unit_forget_bias is None:
                unit_forget_bias = True
            network_parameters.unit_forget_bias = unit_forget_bias

            if activation_funcs is None:
                activation_funcs = [ActivationFunc.TANH, ActivationFunc.LEAKY_RELU] \
                    if n_hidden_lstm_layers == 2 else [ActivationFunc.TANH] * n_hidden_lstm_layers
            network_parameters.activation_funcs = activation_funcs

            if rec_activation_funcs is None:
                rec_activation_funcs = ActivationFunc.SIGMOID
            network_parameters.rec_activation_funcs = rec_activation_funcs

            if dropout is None:
                dropout = 0
            network_parameters.dropout = dropout

            if rec_dropout is None:
                rec_dropout = 0
            network_parameters.rec_dropout = rec_dropout

            if kernel_l1_regularizer is None:
                kernel_l1_regularizer = 0.01
            network_parameters.kernel_l1_regularizer = kernel_l1_regularizer

            if bias_l1_regularizer is None:
                bias_l1_regularizer = 0.01
            network_parameters.bias_l1_regularizer = bias_l1_regularizer

            if recurrent_l1_regularizer is None:
                recurrent_l1_regularizer = 0.01
            network_parameters.recurrent_l1_regularizer = recurrent_l1_regularizer

            if activity_l1_regularizer is None:
                activity_l1_regularizer = 0.01
            network_parameters.activity_l1_regularizer = activity_l1_regularizer

            if kernel_l2_regularizer is None:
                kernel_l2_regularizer = 0.01
            network_parameters.kernel_l2_regularizer = kernel_l2_regularizer

            if bias_l2_regularizer is None:
                bias_l2_regularizer = 0.01
            network_parameters.bias_l2_regularizer = bias_l2_regularizer

            if recurrent_l2_regularizer is None:
                recurrent_l2_regularizer = 0.01
            network_parameters.recurrent_l2_regularizer = recurrent_l2_regularizer

            if activity_l2_regularizer is None:
                activity_l2_regularizer = 0.01
            network_parameters.activity_l2_regularizer = activity_l2_regularizer

            if stateful is None:
                stateful = False
            network_parameters.stateful = stateful

            if go_backwards is None:
                go_backwards = False
            network_parameters.go_backwards = go_backwards

            if dense_instead_lstm_on_out is None:
                dense_instead_lstm_on_out = False
            network_parameters.dense_instead_lstm_on_out = dense_instead_lstm_on_out

            if regression_out_activation_func is None:
                regression_out_activation_func = ActivationFunc.LINEAR
            network_parameters.regression_out_activation_func = regression_out_activation_func

            if binary_out_activation_func is None:
                binary_out_activation_func = ActivationFunc.SIGMOID
            network_parameters.binary_out_activation_func = binary_out_activation_func

            if clip_norm_instead_of_value is None:
                clip_norm_instead_of_value = False
            network_parameters.clip_norm_instead_of_value = clip_norm_instead_of_value

            if clip_norm is None:
                clip_norm = 1.0
            network_parameters.clip_norm = clip_norm

            if clip_value is None:
                clip_value = .5
            network_parameters.clip_value = clip_value

            if batch_size is None:
                batch_size = 5
            network_parameters.batch_size = batch_size

            if optimizer is None:
                optimizer = Optimizer.RMSPROP
            network_parameters.optimizer = optimizer

            if patience_epochs_stop is None:
                patience_epochs_stop = 10
            network_parameters.patience_epochs_stop = patience_epochs_stop

            if patience_epochs_reduce is None:
                patience_epochs_reduce = 10
            network_parameters.patience_epochs_reduce = patience_epochs_reduce

            if reduce_factor is None:
                reduce_factor = .1
            network_parameters.reduce_factor = reduce_factor

            if shuffle is None:
                shuffle = False
            network_parameters.shuffle = shuffle

            if loss is None:
                loss = Loss.MEAN_SQUARED_ERROR
            network_parameters.loss = loss

            if leaky_relu_alpha is None:
                leaky_relu_alpha = 0.3
            network_parameters.leaky_relu_alpha = leaky_relu_alpha
        self.network = network_parameters

        self.validate()

        self.uuid = None
        self.misc_uuid = None
        self.dataset_uuid = None
        self.enricher_uuid = None
        self.pca_uuid = None
        self.network_uuid = None
        self.genAndSetUuids(uuid, misc_uuid, dataset_uuid, enricher_uuid, pca_uuid, network_uuid)

    def validate_misc(self) -> None:
        pred_threshold = 0 <= self.pred_threshold <= 1
        if not pred_threshold:
            raise ValueError(f'Invalid pred_threshold: ({self.pred_threshold}')

    def validate_dataset(self) -> None:
        train_ratio = 0 <= self.dataset.train_ratio < 1
        if not train_ratio:
            raise ValueError(f'Invalid train_ratio: ({self.dataset.train_ratio}')
        validation_ratio = 0 <= self.dataset.validation_ratio < 1
        if not validation_ratio:
            raise ValueError(f'Invalid validation_ratio: ({self.dataset.validation_ratio}')
        ratios = (self.dataset.train_ratio + self.dataset.validation_ratio) <= 1 and \
                 self.dataset.train_ratio >= self.dataset.validation_ratio
        if not ratios:
            raise ValueError(f'Invalid ratios: ({self.dataset.train_ratio},{self.dataset.validation_ratio}')
        # n_features, ignore
        if type(self.dataset.norm_range) is list:
            self.dataset.norm_range = tuple(self.dataset.norm_range)
        norm_range = len(self.dataset.norm_range) == 2 and \
                     -1 <= self.dataset.norm_range[0] <= 1 and \
                     -1 <= self.dataset.norm_range[1] <= 1 and type(self.dataset.norm_range) is tuple
        if not norm_range:
            raise ValueError(f'Invalid norm_range: ({self.dataset.norm_range}')

    def validate_enricher(self) -> None:
        fibonacci_seq_size = self.enricher.fibonacci_seq_size > 2
        if not fibonacci_seq_size:
            raise ValueError(f'Invalid fibonacci_seq_size: ({self.enricher.fibonacci_seq_size}')
        fast_average_window = self.enricher.fast_average_window > 2
        if not fast_average_window:
            raise ValueError(f'Invalid fast_average_window: ({self.enricher.fast_average_window}')
        slow_average_window = self.enricher.slow_average_window > 2
        if not slow_average_window:
            raise ValueError(f'Invalid slow_average_window: ({self.enricher.slow_average_window}')

    def validate_pca(self) -> None:
        if type(self.pca.pca_norm_range) is list:
            self.pca.pca_norm_range = tuple(self.pca.pca_norm_range)
        pca_norm_range = len(self.pca.pca_norm_range) == 2 and \
                         -1 <= self.pca.pca_norm_range[0] <= 1 and \
                         -1 <= self.pca.pca_norm_range[1] <= 1 and type(self.pca.pca_norm_range) is tuple
        if not pca_norm_range:
            raise ValueError(f'Invalid pca_norm_range: ({self.pca.pca_norm_range}')
        knee_sensibility = self.pca.knee_sensibility > 1
        if not knee_sensibility:
            raise ValueError(f'Invalid knee_sensibility: ({self.pca.knee_sensibility}')
        self.pca.kernel_pca_type = self.pca.kernel_pca_type.lower()
        kernel = self.pca.kernel_pca_type in ('linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed')
        if not kernel:
            raise ValueError(f'Invalid kernel: ({self.pca.kernel_pca_type}')

    def validate_network(self) -> None:
        backward_samples = self.network.backward_samples > 1
        if not backward_samples:
            raise ValueError(f'Invalid backward_samples: ({self.network.backward_samples}')
        forward_samples = self.network.forward_samples > 0
        if not forward_samples:
            raise ValueError(f'Invalid forward_samples: ({self.network.forward_samples}')
        max_epochs = self.network.max_epochs > 0
        if not max_epochs:
            raise ValueError(f'Invalid max_epochs: ({self.network.max_epochs}')
        patience_epochs_stop = self.network.patience_epochs_stop >= 0
        if not patience_epochs_stop:
            raise ValueError(f'Invalid patience_epochs_stop: ({self.network.patience_epochs_stop}')
        patience_epochs_reduce = self.network.patience_epochs_reduce >= 0
        if not patience_epochs_reduce:
            raise ValueError(f'Invalid patience_epochs_reduce: ({self.network.patience_epochs_reduce}')
        # patience cannot be higher than epochs
        self.network.patience_epochs_stop = min(self.network.patience_epochs_stop, self.network.max_epochs)
        self.network.patience_epochs_reduce = min(self.network.patience_epochs_reduce, self.network.max_epochs)
        reduce_factor = 0 <= self.network.reduce_factor <= 1
        if not reduce_factor:
            raise ValueError(f'Invalid reduce_factor: ({self.network.reduce_factor}')
        n_hidden_lstm_layers = self.network.n_hidden_lstm_layers >= 0
        if not n_hidden_lstm_layers:
            raise ValueError(f'Invalid n_hidden_lstm_layers: ({self.network.n_hidden_lstm_layers}')
        if self.network.batch_size is None:
            self.network.batch_size = 0
        batch_size = self.network.batch_size >= 0
        if not batch_size:
            raise ValueError(f'Invalid batch_size: ({self.network.batch_size}')
        if self.network.batch_size % 2 != 0:
            self.network.batch_size += 1  # only even batch_sizes
        leaky_relu_alpha = 0 <= self.network.leaky_relu_alpha <= 1
        if not leaky_relu_alpha:
            raise ValueError(f'Invalid range for leaky_relu_alpha: ({self.network.leaky_relu_alpha}')

        if type(self.network.layer_sizes) in getNumericTypes():
            self.network.layer_sizes = [self.network.layer_sizes] * self.network.n_hidden_lstm_layers
        if type(self.network.use_bias) is bool:
            self.network.use_bias = [self.network.use_bias] * self.network.n_hidden_lstm_layers
        if type(self.network.unit_forget_bias) is bool:
            self.network.unit_forget_bias = [self.network.unit_forget_bias] * self.network.n_hidden_lstm_layers
        if type(self.network.activation_funcs) in (int, ActivationFunc):
            self.network.activation_funcs = [self.network.activation_funcs] * self.network.n_hidden_lstm_layers
        if type(self.network.rec_activation_funcs) in (int, ActivationFunc):
            self.network.rec_activation_funcs = [self.network.rec_activation_funcs] * self.network.n_hidden_lstm_layers
        if type(self.network.dropout) in getNumericTypes():
            self.network.dropout = [self.network.dropout] * self.network.n_hidden_lstm_layers
        if type(self.network.rec_dropout) in getNumericTypes():
            self.network.rec_dropout = [self.network.rec_dropout] * self.network.n_hidden_lstm_layers
        if type(self.network.kernel_l1_regularizer) in getNumericTypes():
            self.network.kernel_l1_regularizer = [self.network.kernel_l1_regularizer] * (
                    self.network.n_hidden_lstm_layers + 1)
        if type(self.network.bias_l1_regularizer) in getNumericTypes():
            self.network.bias_l1_regularizer = [self.network.bias_l1_regularizer] * (
                    self.network.n_hidden_lstm_layers + 1)
        if type(self.network.recurrent_l1_regularizer) in getNumericTypes():
            self.network.recurrent_l1_regularizer = [self.network.recurrent_l1_regularizer] * (
                    self.network.n_hidden_lstm_layers + 1)
        if type(self.network.activity_l1_regularizer) in getNumericTypes():
            self.network.activity_l1_regularizer = [self.network.activity_l1_regularizer] * (
                    self.network.n_hidden_lstm_layers + 1)
        if type(self.network.kernel_l2_regularizer) in getNumericTypes():
            self.network.kernel_l2_regularizer = [self.network.kernel_l2_regularizer] * (
                    self.network.n_hidden_lstm_layers + 1)
        if type(self.network.bias_l2_regularizer) in getNumericTypes():
            self.network.bias_l2_regularizer = [self.network.bias_l2_regularizer] * (
                    self.network.n_hidden_lstm_layers + 1)
        if type(self.network.recurrent_l2_regularizer) in getNumericTypes():
            self.network.recurrent_l2_regularizer = [self.network.recurrent_l2_regularizer] * (
                    self.network.n_hidden_lstm_layers + 1)
        if type(self.network.activity_l2_regularizer) in getNumericTypes():
            self.network.activity_l2_regularizer = [self.network.activity_l2_regularizer] * (
                    self.network.n_hidden_lstm_layers + 1)

        if type(self.network.go_backwards) is bool:
            self.network.go_backwards = [self.network.go_backwards] * self.network.n_hidden_lstm_layers

        if len(self.network.dropout) != self.network.n_hidden_lstm_layers:
            raise ValueError(f'Wrong dropout_values array size, should be {self.network.n_hidden_lstm_layers} '
                             f'instead of {len(self.network.dropout)}')
        if len(self.network.layer_sizes) != self.network.n_hidden_lstm_layers and not (
                self.network.layer_sizes[0] == self.network.backward_samples
                and len(self.network.layer_sizes) == self.network.n_hidden_lstm_layers + 1):
            raise ValueError(f'Wrong layer_sizes array size, should be {self.network.n_hidden_lstm_layers}')
        if len(self.network.activation_funcs) != self.network.n_hidden_lstm_layers:
            raise ValueError(f'Wrong activation_functions array size, should be {self.network.n_hidden_lstm_layers}')
        if len(self.network.rec_activation_funcs) != self.network.n_hidden_lstm_layers:
            raise ValueError(
                f'Wrong recurrent_activation_functions array size, should be {self.network.n_hidden_lstm_layers}')
        if len(self.network.use_bias) != self.network.n_hidden_lstm_layers:
            raise ValueError(f'Wrong bias array size, should be {self.network.n_hidden_lstm_layers}')
        if len(self.network.unit_forget_bias) != self.network.n_hidden_lstm_layers:
            raise ValueError(f'Wrong unit_forget_bias array size, should be {self.network.n_hidden_lstm_layers}')
        if len(self.network.go_backwards) != self.network.n_hidden_lstm_layers:
            raise ValueError(f'Wrong go_backwards array size, should be {self.network.n_hidden_lstm_layers}')
        if len(self.network.rec_dropout) != self.network.n_hidden_lstm_layers:
            raise ValueError(
                f'Wrong rec_dropout array size, should be {self.network.n_hidden_lstm_layers}')
        if len(self.network.kernel_l1_regularizer) != self.network.n_hidden_lstm_layers + 1:
            raise ValueError(
                f'Wrong kernel_l1_regularizer array size, should be {self.network.n_hidden_lstm_layers + 1}')
        if len(self.network.bias_l1_regularizer) != self.network.n_hidden_lstm_layers + 1:
            raise ValueError(
                f'Wrong bias_l1_regularizer array size, should be {self.network.n_hidden_lstm_layers + 1}')
        if len(self.network.recurrent_l1_regularizer) != self.network.n_hidden_lstm_layers + 1:
            raise ValueError(
                f'Wrong recurrent_l1_regularizer array size, should be {self.network.n_hidden_lstm_layers + 1}')
        if len(self.network.activity_l1_regularizer) != self.network.n_hidden_lstm_layers + 1:
            raise ValueError(
                f'Wrong activity_l1_regularizer array size, should be {self.network.n_hidden_lstm_layers + 1}')
        if len(self.network.kernel_l2_regularizer) != self.network.n_hidden_lstm_layers + 1:
            raise ValueError(
                f'Wrong kernel_l2_regularizer array size, should be {self.network.n_hidden_lstm_layers + 1}')
        if len(self.network.bias_l2_regularizer) != self.network.n_hidden_lstm_layers + 1:
            raise ValueError(
                f'Wrong bias_l2_regularizer array size, should be {self.network.n_hidden_lstm_layers + 1}')
        if len(self.network.recurrent_l2_regularizer) != self.network.n_hidden_lstm_layers + 1:
            raise ValueError(
                f'Wrong recurrent_l2_regularizer array size, should be {self.network.n_hidden_lstm_layers + 1}')
        if len(self.network.activity_l2_regularizer) != self.network.n_hidden_lstm_layers + 1:
            raise ValueError(
                f'Wrong activity_l2_regularizer array size, should be {self.network.n_hidden_lstm_layers + 1}')

        for i in range(len(self.network.activation_funcs)):
            if type(self.network.activation_funcs[i]) in getNumericTypes():
                self.network.activation_funcs[i] = ActivationFunc(self.network.activation_funcs[i])
        for i in range(len(self.network.rec_activation_funcs)):
            if type(self.network.rec_activation_funcs[i]) in getNumericTypes():
                self.network.rec_activation_funcs[i] = ActivationFunc(self.network.rec_activation_funcs[i])

        if type(self.network.regression_out_activation_func) in getNumericTypes():
            self.network.regression_out_activation_func = ActivationFunc(self.network.regression_out_activation_func)
        if type(self.network.binary_out_activation_func) in getNumericTypes():
            self.network.binary_out_activation_func = ActivationFunc(self.network.binary_out_activation_func)
        if type(self.network.optimizer) in getNumericTypes():
            self.network.optimizer = Optimizer(self.network.optimizer)
        if type(self.network.loss) in getNumericTypes():
            self.network.loss = Loss(self.network.loss)

        if self.network.stateful and self.network.batch_size == 0:
            self.network.batch_size = 1  # batch size must be one for stateful
        if len(self.network.layer_sizes) == self.network.n_hidden_lstm_layers:
            self.network.layer_sizes.insert(0, self.network.backward_samples)

    def validate(self) -> None:
        self.validate_misc()
        self.validate_dataset()
        self.validate_enricher()
        self.validate_pca()
        self.validate_network()

    def setDatasetFilename(self, filename: str) -> None:
        filename = getBasename(filename)
        self.dataset_filename = filename

    def setFilenameAndLoadAmountFeatures(self, filename: str) -> None:
        self.setDatasetFilename(filename)
        n = self.loadAmountOfFeatures()
        if n is not None:
            self.dataset.n_features = n

    def loadAmountOfFeatures(self) -> Optional[int]:
        if self.dataset_filename is not None:
            # todo fix this workaround
            from transformer import loadAmountOfFeaturesFromFile
            return loadAmountOfFeaturesFromFile(self.dataset_filename)

    def toString(self, show_uuids=True) -> str:
        if show_uuids:
            out = f"""uuid: {self.uuid},
misc_uuid: {self.misc_uuid},  
dataset_uuid: {self.dataset_uuid},  
enricher_uuid: {self.enricher_uuid},  
pca_uuid: {self.pca_uuid},  
network_uuid: {self.network_uuid},  
"""
        else:
            out = ''
        misc = self.toStringMisc()
        dataset = self.toStringDataset()
        enricher = self.toStringEnricher()
        pca = self.toStringPCA()
        network = self.toStringNetwork()

        tab = "\t"
        out = f'Hyperparameters:\n{textwrap.indent(out, tab)}{textwrap.indent(misc, tab)}\n\tDataset:\n' \
              f'{textwrap.indent(dataset, tab * 2)}\n\tEnricher:\n{textwrap.indent(enricher, tab * 2)}\n\tPCA:\n' \
              f'{textwrap.indent(pca, tab * 2)}\n\tNetwork:\n{textwrap.indent(network, tab * 2)} '
        return out

    def toStringMisc(self) -> str:
        return f"""name: {self.name},
dataset_filename: {self.dataset_filename},  
pred_threshold: {self.pred_threshold},  
"""

    def toStringDataset(self, ignore_n_feats: bool = False) -> str:
        return f"""train_ratio: {self.dataset.train_ratio},
validation_ratio: {self.dataset.validation_ratio},   
normalize: {self.dataset.normalize},  
normalize_prediction_feat: {self.dataset.normalize_prediction_feat},  
norm_range: {self.dataset.norm_range},  
""" + ("n_features: {self.dataset.n_features}, \n" if not ignore_n_feats else "")

    def toStringEnricher(self) -> str:
        return f"""price_ratios_on: {self.enricher.price_ratios_on},
price_delta_on: {self.enricher.price_delta_on},  
price_averages_on: {self.enricher.price_averages_on},  
ta_lib_on: {self.enricher.ta_lib_on},  
fibonacci_seq_size: {self.enricher.fibonacci_seq_size},  
fast_average_window: {self.enricher.fast_average_window},  
slow_average_window: {self.enricher.slow_average_window},  
"""

    def toStringPCA(self) -> str:
        return f"""use_kernel_pca: {self.pca.use_kernel_pca},
kernel_pca_type: {self.pca.kernel_pca_type},  
pca_norm_range: {self.pca.pca_norm_range},  
knee_sensibility: {self.pca.knee_sensibility},  
"""

    def toStringNetwork(self) -> str:
        return f"""backward_samples: {self.network.backward_samples},
forward_samples: {self.network.forward_samples},  
max_epochs: {self.network.max_epochs},  
n_hidden_lstm_layers: {self.network.n_hidden_lstm_layers},  
layer_sizes: {self.network.layer_sizes},  
use_bias: {self.network.use_bias},  
unit_forget_bias: {self.network.unit_forget_bias},  
activation_funcs: {self.network.activation_funcs},  
rec_activation_funcs: {self.network.rec_activation_funcs},  
dropout: {self.network.dropout},  
rec_dropout: {self.network.rec_dropout},  
kernel_l1_regularizer: {self.network.kernel_l1_regularizer},  
bias_l1_regularizer: {self.network.bias_l1_regularizer},  
recurrent_l1_regularizer: {self.network.recurrent_l1_regularizer},  
activity_l1_regularizer: {self.network.activity_l1_regularizer},  
kernel_l2_regularizer: {self.network.kernel_l2_regularizer},  
bias_l2_regularizer: {self.network.bias_l2_regularizer},  
recurrent_l2_regularizer: {self.network.recurrent_l2_regularizer},  
activity_l2_regularizer: {self.network.activity_l2_regularizer},  
stateful: {self.network.stateful},  
go_backwards: {self.network.go_backwards},  
dense_instead_lstm_on_out: {self.network.dense_instead_lstm_on_out},  
batch_size: {self.network.batch_size},  
regression_out_activation_func: {self.network.regression_out_activation_func},  
binary_out_activation_func: {self.network.binary_out_activation_func},  
clip_norm_instead_of_value: {self.network.clip_norm_instead_of_value},  
clip_norm: {self.network.clip_norm},  
clip_value: {self.network.clip_value},  
optimizer: {self.network.optimizer},  
patience_epochs_stop: {self.network.patience_epochs_stop},  
patience_epochs_reduce: {self.network.patience_epochs_reduce},  
reduce_factor: {self.network.reduce_factor},  
shuffle: {self.network.shuffle},  
loss: {self.network.loss},  
leaky_relu_alpha: {self.network.leaky_relu_alpha},  
"""

    def genUuids(self) -> tuple[str, str, str, str, str, str]:
        misc = hashStr(self.toStringMisc())
        dataset = hashStr(self.toStringDataset())
        enricher = hashStr(self.toStringEnricher())
        pca = hashStr(self.toStringPCA())
        network = hashStr(self.toStringNetwork())

        whole = hashStr(f"{misc}{dataset}{enricher}{pca}{network}")
        return whole, misc, dataset, enricher, pca, network

    def genDatasetEnricherPcaUuid(self) -> str:
        return hashStr(f"{self.toStringDataset(ignore_n_feats=False)}{self.enricher_uuid}{self.pca_uuid}")

    def refreshUuids(self) -> None:
        self.genAndSetUuids()

    def genAndSetUuids(self, whole: Optional[str] = None, misc: Optional[str] = None, dataset: Optional[str] = None,
                       enricher: Optional[str] = None, pca: Optional[str] = None,
                       network: Optional[str] = None) -> None:
        new_whole, new_misc, new_dataset, new_enricher, new_pca, new_network = self.genUuids()
        if whole is None:
            whole = new_whole
        if misc is None:
            misc = new_misc
        if dataset is None:
            dataset = new_dataset
        if enricher is None:
            enricher = new_enricher
        if pca is None:
            pca = new_pca
        if network is None:
            network = new_network
        self.misc_uuid = misc
        self.dataset_uuid = dataset
        self.enricher_uuid = enricher
        self.pca_uuid = pca
        self.network_uuid = network
        self.uuid = whole

    def getConfigFilepath(self, subdir: str = '') -> str:
        filename = re.sub(r'\W+', '',
                          self.name.replace(' ', '_').lower()) + '-' if self.name is not None else ''
        filename = f'{filename}{self.uuid}.json'
        filepath = pathJoin(HYPERPARAMETERS_DIR, subdir, filename)
        return filepath

    def copy(self) -> Hyperparameters:
        return Hyperparameters.loadFromDict(self.toDict())

    def toDict(self) -> dict:
        hyperparameters_dict = self.__dict__.copy()
        hyperparameters_dict['__type__'] = 'Hyperparameters'
        hyperparameters_dict['dataset'] = hyperparameters_dict['dataset'].__dict__
        hyperparameters_dict['enricher'] = hyperparameters_dict['enricher'].__dict__
        hyperparameters_dict['pca'] = hyperparameters_dict['pca'].__dict__
        hyperparameters_dict['network'] = hyperparameters_dict['network'].__dict__
        return hyperparameters_dict

    def saveJson(self, filepath: Optional[str] = None, subdir: str = '') -> str:
        if filepath is None:
            filepath = self.getConfigFilepath(subdir)
            hyperparameters_dict = self.toDict()
        saveJson(hyperparameters_dict, filepath)
        return filepath

    @staticmethod
    def loadFromDict(obj: dict) -> Hyperparameters:
        return Hyperparameters(
            # uuid
            # uuid=obj.get('uuid', None),
            # misc_uuid=obj.get('misc_uuid', None),
            # dataset_uuid=obj.get('dataset_uuid', None),
            # enricher_uuid=obj.get('enricher_uuid', None),
            # pca_uuid=obj.get('pca_uuid', None),
            # network_uuid=obj.get('network_uuid', None),
            # misc
            name=obj.get('name', None),
            dataset_filename=obj.get('dataset_filename', None),
            pred_threshold=obj.get('pred_threshold', None),
            # dataset
            train_ratio=obj.get('dataset', obj).get('train_ratio', None),
            validation_ratio=obj.get('dataset', obj).get('validation_ratio', None),
            n_features=obj.get('dataset', obj).get('n_features', None),
            normalize=obj.get('dataset', obj).get('normalize', None),
            normalize_prediction_feat=obj.get('dataset', obj).get('normalize_prediction_feat', None),
            norm_range=obj.get('dataset', obj).get('norm_range', None),
            # enricher
            price_ratios_on=obj.get('enricher', obj).get('price_ratios_on', None),
            price_delta_on=obj.get('enricher', obj).get('price_delta_on', None),
            price_averages_on=obj.get('enricher', obj).get('price_averages_on', None),
            ta_lib_on=obj.get('enricher', obj).get('ta_lib_on', None),
            fibonacci_seq_size=obj.get('enricher', obj).get('fibonacci_seq_size', None),
            fast_average_window=obj.get('enricher', obj).get('fast_average_window', None),
            slow_average_window=obj.get('enricher', obj).get('slow_average_window', None),
            # pca
            use_kernel_pca=obj.get('pca', obj).get('use_kernel_pca', None),
            kernel_pca_type=obj.get('pca', obj).get('kernel_pca_type', None),
            pca_norm_range=obj.get('pca', obj).get('pca_norm_range', None),
            knee_sensibility=obj.get('pca', obj).get('knee_sensibility', None),
            # network
            backward_samples=obj.get('network', obj).get('backward_samples', None),
            forward_samples=obj.get('network', obj).get('forward_samples', None),
            max_epochs=obj.get('network', obj).get('max_epochs', None),
            n_hidden_lstm_layers=obj.get('network', obj).get('n_hidden_lstm_layers', None),
            layer_sizes=obj.get('network', obj).get('layer_sizes', None),
            use_bias=obj.get('network', obj).get('use_bias', None),
            unit_forget_bias=obj.get('network', obj).get('unit_forget_bias', None),
            activation_funcs=obj.get('network', obj).get('activation_funcs', None),
            rec_activation_funcs=obj.get('network', obj).get('rec_activation_funcs', None),
            dropout=obj.get('network', obj).get('dropout', None),
            rec_dropout=obj.get('network', obj).get('rec_dropout', None),
            kernel_l1_regularizer=obj.get('network', obj).get('kernel_l1_regularizer', None),
            bias_l1_regularizer=obj.get('network', obj).get('bias_l1_regularizer', None),
            recurrent_l1_regularizer=obj.get('network', obj).get('recurrent_l1_regularizer', None),
            activity_l1_regularizer=obj.get('network', obj).get('activity_l1_regularizer', None),
            kernel_l2_regularizer=obj.get('network', obj).get('kernel_l2_regularizer', None),
            bias_l2_regularizer=obj.get('network', obj).get('bias_l2_regularizer', None),
            recurrent_l2_regularizer=obj.get('network', obj).get('recurrent_l2_regularizer', None),
            activity_l2_regularizer=obj.get('network', obj).get('activity_l2_regularizer', None),
            stateful=obj.get('network', obj).get('stateful', None),
            go_backwards=obj.get('network', obj).get('go_backwards', None),
            dense_instead_lstm_on_out=obj.get('network', obj).get('dense_instead_lstm_on_out', None),
            batch_size=obj.get('network', obj).get('batch_size', None),
            regression_out_activation_func=obj.get('network', obj).get('regression_out_activation_func', None),
            binary_out_activation_func=obj.get('network', obj).get('binary_out_activation_func', None),
            clip_norm_instead_of_value=obj.get('network', obj).get('clip_norm_instead_of_value', None),
            clip_norm=obj.get('network', obj).get('clip_norm', None),
            clip_value=obj.get('network', obj).get('clip_value', None),
            optimizer=obj.get('network', obj).get('optimizer', None),
            patience_epochs_stop=obj.get('network', obj).get('patience_epochs_stop', None),
            patience_epochs_reduce=obj.get('network', obj).get('patience_epochs_reduce', None),
            reduce_factor=obj.get('network', obj).get('reduce_factor', None),
            shuffle=obj.get('network', obj).get('shuffle', None),
            loss=obj.get('network', obj).get('loss', None),
            leaky_relu_alpha=obj.get('network', obj).get('leaky_relu_alpha', None),
        )

    @staticmethod
    def jsonDecoder(obj: dict) -> Union[Hyperparameters, dict]:
        if '__type__' in obj and obj['__type__'] == 'Hyperparameters':
            return Hyperparameters.loadFromDict(obj)
        return obj

    @staticmethod
    def enrichSearchSpace(search_space: SearchSpace) -> SearchSpace:
        enriched_space = SearchSpace(name=search_space.name)
        genome_map = {}
        # TODO make search_const_or_die and search_const_or_check(later)
        fields = [
            dict(name='name', type_match=SearchSpace.Type.CONSTANT, action='dont_track'),
            dict(name='dataset_filename', type_match=SearchSpace.Type.CONSTANT, action='dont_track'),
            dict(name='pred_threshold', type_match=SearchSpace.Type.FLOAT, fallback='search_const', action='track'),
            dict(name='train_ratio', type_match=SearchSpace.Type.FLOAT, fallback='search_const', action='track'),
            dict(name='validation_ratio', type_match=SearchSpace.Type.FLOAT, fallback='search_const', action='track'),
            dict(name='n_features', type_match=SearchSpace.Type.CONSTANT, action='ignore',
                 comments='Here just to have all fields!'),
            dict(name='normalize', type_match=SearchSpace.Type.BOOLEAN, fallback='search_const', action='track'),
            dict(name='normalize_prediction_feat', type_match=SearchSpace.Type.BOOLEAN, fallback='search_const',
                 action='track'),
            dict(name='norm_range', action='ignore', comments='Here just to have all fields!'),
            dict(name='price_ratios_on', type_match=SearchSpace.Type.BOOLEAN, fallback='search_const', action='track'),
            dict(name='price_delta_on', type_match=SearchSpace.Type.BOOLEAN, fallback='search_const', action='track'),
            dict(name='price_averages_on', type_match=SearchSpace.Type.BOOLEAN, fallback='search_const',
                 action='track'),
            dict(name='ta_lib_on', type_match=SearchSpace.Type.BOOLEAN, fallback='search_const', action='track'),
            dict(name='fibonacci_seq_size', type_match=SearchSpace.Type.INT, fallback='search_const', action='track'),
            dict(name='fast_average_window', type_match=SearchSpace.Type.INT, fallback='search_const', action='track'),
            dict(name='slow_average_window', type_match=SearchSpace.Type.INT, fallback='search_const', action='track'),
            dict(name='use_kernel_pca', action='ignore', comments='Here just to have all fields!'),
            dict(name='kernel_pca_type', action='ignore', comments='Here just to have all fields!'),
            dict(name='pca_norm_range', action='ignore', comments='Here just to have all fields!'),
            dict(name='knee_sensibility', action='ignore', comments='Here just to have all fields!'),
            dict(name='backward_samples', type_match=SearchSpace.Type.INT, action='track', mandatory=True),
            dict(name='forward_samples', type_match=SearchSpace.Type.INT, action='track', mandatory=True),
            dict(name='max_epochs', type_match=SearchSpace.Type.INT, action='track', mandatory=True),
            dict(name='n_hidden_lstm_layers', type_match=SearchSpace.Type.INT, action='track_and_save', mandatory=True),
            dict(name='layer_sizes', type_match=SearchSpace.Type.INT, action='track_list',
                 list_size='n_hidden_lstm_layers', mandatory=True),
            dict(name='use_bias', type_match=SearchSpace.Type.BOOLEAN, action='track_list',
                 list_size='n_hidden_lstm_layers', mandatory=True),
            dict(name='unit_forget_bias', type_match=SearchSpace.Type.BOOLEAN, action='track_list',
                 list_size='n_hidden_lstm_layers', mandatory=True),

            dict(name='activation_funcs', type_match=SearchSpace.Type.INT, fallback='check_later', action='track_list',
                 list_size='n_hidden_lstm_layers', mandatory=True),
            dict(name='activation_funcs', type_match=SearchSpace.Type.CHOICE, fallback='check_later',
                 action='track_list', list_size='n_hidden_lstm_layers', mandatory=True),

            dict(name='rec_activation_funcs', type_match=SearchSpace.Type.INT, fallback='check_later',
                 action='track_list', list_size='n_hidden_lstm_layers', mandatory=True),
            dict(name='rec_activation_funcs', type_match=SearchSpace.Type.CHOICE, fallback='check_later',
                 action='track_list', list_size='n_hidden_lstm_layers', mandatory=True),

            dict(name='dropout', type_match=SearchSpace.Type.FLOAT, fallback='search_const', action='track_list',
                 list_size='n_hidden_lstm_layers', mandatory=True),
            dict(name='rec_dropout', type_match=SearchSpace.Type.FLOAT, fallback='search_const', action='track_list',
                 list_size='n_hidden_lstm_layers', mandatory=True),
            dict(name='kernel_l1_regularizer', type_match=SearchSpace.Type.FLOAT, fallback='search_const',
                 action='track_list', list_size='n_hidden_lstm_layers', increase_list_by=1),
            dict(name='bias_l1_regularizer', type_match=SearchSpace.Type.FLOAT, fallback='search_const',
                 action='track_list', list_size='n_hidden_lstm_layers', increase_list_by=1),
            dict(name='recurrent_l1_regularizer', type_match=SearchSpace.Type.FLOAT, fallback='search_const',
                 action='track_list', list_size='n_hidden_lstm_layers', increase_list_by=1),
            dict(name='activity_l1_regularizer', type_match=SearchSpace.Type.FLOAT, fallback='search_const',
                 action='track_list', list_size='n_hidden_lstm_layers', increase_list_by=1),
            dict(name='kernel_l2_regularizer', type_match=SearchSpace.Type.FLOAT, fallback='search_const',
                 action='track_list', list_size='n_hidden_lstm_layers', increase_list_by=1),
            dict(name='bias_l2_regularizer', type_match=SearchSpace.Type.FLOAT, fallback='search_const',
                 action='track_list', list_size='n_hidden_lstm_layers', increase_list_by=1),
            dict(name='recurrent_l2_regularizer', type_match=SearchSpace.Type.FLOAT, fallback='search_const',
                 action='track_list', list_size='n_hidden_lstm_layers', increase_list_by=1),
            dict(name='activity_l2_regularizer', type_match=SearchSpace.Type.FLOAT, fallback='search_const',
                 action='track_list', list_size='n_hidden_lstm_layers', increase_list_by=1),
            dict(name='stateful', type_match=SearchSpace.Type.BOOLEAN, action='track', mandatory=True),
            dict(name='go_backwards', type_match=SearchSpace.Type.BOOLEAN, fallback='search_const', action='track_list',
                 list_size='n_hidden_lstm_layers'),
            dict(name='dense_instead_lstm_on_out', type_match=SearchSpace.Type.BOOLEAN, action='track', mandatory=True),
            dict(name='batch_size', type_match=SearchSpace.Type.INT, action='track', mandatory=True),
            dict(name='regression_out_activation_func', action='ignore', comments='Here just to have all fields!'),
            dict(name='binary_out_activation_func', action='ignore', comments='Here just to have all fields!'),
            dict(name='clip_norm_instead_of_value', type_match=SearchSpace.Type.BOOLEAN, fallback='search_const',
                 action='track'),
            dict(name='clip_norm', type_match=SearchSpace.Type.FLOAT, fallback='search_const', action='track'),
            dict(name='clip_value', type_match=SearchSpace.Type.FLOAT, fallback='search_const', action='track'),

            dict(name='optimizer', type_match=SearchSpace.Type.INT, fallback='check_later', action='track'),
            dict(name='optimizer', type_match=SearchSpace.Type.CHOICE, fallback='check_later', action='track'),

            dict(name='patience_epochs_stop', type_match=SearchSpace.Type.INT, fallback='search_const', action='track'),
            dict(name='patience_epochs_reduce', type_match=SearchSpace.Type.INT, fallback='search_const',
                 action='track'),
            dict(name='reduce_factor', type_match=SearchSpace.Type.FLOAT, fallback='search_const', action='track'),
            dict(name='shuffle', type_match=SearchSpace.Type.BOOLEAN, fallback='search_const', action='track'),

            dict(name='loss', type_match=SearchSpace.Type.INT, fallback='check_later', action='track'),
            dict(name='loss', type_match=SearchSpace.Type.CHOICE, fallback='check_later', action='track'),

            dict(name='leaky_relu_alpha', type_match=SearchSpace.Type.FLOAT, fallback='search_const', action='track'),
        ]

        mandatory_fields = set()
        optional_fields = set()
        stored_values = {}
        for field in fields:
            name = field['name']
            action = field['action']
            type_match = field.get('type_match', None)
            fallback = field.get('fallback', 'fatal')
            mandatory = field.get('mandatory', False)
            list_size = field.get('list_size', None)
            increase_list_by = field.get('increase_list_by', 0)
            if not mandatory and fallback != 'check_later':
                optional_fields.add(name)
            if action in ('dont_track', 'track', 'track_and_save', 'track_list'):
                if name in search_space:
                    dim = search_space[name]
                    if type_match is None or (type(type_match) in (list, set) and dim.data_type in type_match) \
                            or dim.data_type == type_match:
                        if action != 'track_list':
                            idx = enriched_space.add(dimension=dim)
                        else:
                            idx = None
                        if action in ('track', 'track_and_save'):
                            genome_map[name] = idx
                        if action == 'track_and_save':
                            if dim.data_type == SearchSpace.Type.INT:
                                value = dim.max_value
                            elif dim.data_type == SearchSpace.Type.CONSTANT:
                                value = dim.const
                            elif dim.data_type == SearchSpace.Type.BOOLEAN:
                                value = True
                            else:
                                value = None
                            if value is not None:
                                stored_values[name] = value
                        if action == 'track_list':
                            if list_size is None:
                                fatal(f'Provide the list_size when action is {action}')
                            if type(list_size) in getNumericTypes():
                                pass
                            elif type(list_size) is str:
                                list_size = stored_values[list_size]
                            else:
                                ValueError(f'Invalid list_size type {type(list_size)}')
                            list_size += increase_list_by
                            first_idx = None
                            last_idx = None
                            for i in range(list_size + 1):
                                new_name = f'{name}[{i}]'
                                idx = enriched_space.add(data_type=dim.data_type, min_value=dim.min_value,
                                                         max_value=dim.max_value, choices=dim.choices, const=dim.const,
                                                         name=new_name)
                                if first_idx is None:
                                    first_idx = idx
                                last_idx = idx
                            genome_map[name] = [first_idx, last_idx]

                    else:
                        msg = f'Dimension {name} of {search_space.name} should be {type_match}!'
                        if fallback == 'ignore':
                            pass
                        elif fallback == 'fatal':
                            fatal(msg)
                        elif fallback == 'warn':
                            warn(msg)
                        elif fallback == 'check_later':
                            mandatory_fields.add(name)
                        elif fallback == 'search_const':
                            if dim.data_type == SearchSpace.Type.CONSTANT:
                                idx = enriched_space.add(dimension=dim)
                                if action == 'track':
                                    genome_map[name] = idx
                        else:
                            raise ValueError(f'Unknown fallback: `{fallback}`')
                elif mandatory and action != 'check_later':
                    fatal(f'Dimension {name} of {search_space.name} is mandatory!')
            elif action == 'ignore':
                pass
            else:
                raise ValueError(f'Unknown action: `{action}`')

        # TODO set default values to important fields

        for field in mandatory_fields:
            if field not in enriched_space and f'{field}[{0}]' not in enriched_space:
                fatal(f'Field {field} should be in enriched_space {enriched_space.name}!')

        enriched_space.genome_map = genome_map
        return enriched_space

    @staticmethod
    def parseDna(dna: Union[dict, list, np.ndarray], search_space: SearchSpace) -> Hyperparameters:
        hyperparameters_dict = {}
        array_dna = type(dna) is not dict
        array_features = set()
        if not array_dna:
            hyperparameters_dict['n_features'] = dna.get('n_features', None)
        for feature, index in search_space.genome_map.items():
            if (feature in search_space and search_space[feature].data_type == SearchSpace.Type.CONSTANT) or \
                    (feature[::-1].split('[', 1)[-1][::-1] in search_space and search_space[
                        feature[::-1].split('[', 1)[-1][::-1]].data_type == SearchSpace.Type.CONSTANT):
                continue
            if type(index) in getNumericTypes():
                if array_dna:
                    index = search_space.applyConstantsMask(index)
                    value = dna[index]
                    hyperparameters_dict[feature] = value
                else:
                    hyperparameters_dict[feature] = dna[feature]
            else:
                array_features.add(feature)
                if array_dna:
                    begin, end = search_space.applyConstantsMask(index[0]), search_space.applyConstantsMask(index[1])
                    values = []
                    feature = feature[::-1].split('[', 1)[-1][::-1]
                    for itr in range(begin, end + 1, 1):
                        itr = search_space.applyConstantsMask(itr)
                        values.append(dna[itr])
                    hyperparameters_dict[feature] = values
                else:
                    values = []
                    for itr in range(index[1] - index[0] + 1):
                        values.append(dna[f'{feature}[{itr}]'])
                    hyperparameters_dict[feature] = values
        h_layers = hyperparameters_dict['n_hidden_lstm_layers']
        for array_feature in array_features:
            max_size = h_layers
            if array_feature.endswith('_regularizer'):  # TODO make map to store the size of array features
                max_size += 1
            hyperparameters_dict[array_feature] = hyperparameters_dict[array_feature][:max_size]
        for dimension in search_space:
            if dimension.data_type == SearchSpace.Type.CONSTANT and dimension.name not in hyperparameters_dict:
                hyperparameters_dict[dimension.name] = dimension.const
        return Hyperparameters.loadFromDict(hyperparameters_dict)

    @staticmethod
    def loadJson(filepath: str) -> Hyperparameters:
        return loadJson(filepath, Hyperparameters.jsonDecoder)

    @staticmethod
    def getDefault() -> Hyperparameters:
        if Hyperparameters.DEFAULT is None:
            Hyperparameters.DEFAULT = Hyperparameters()
        return Hyperparameters.DEFAULT


createFolder(HYPERPARAMETERS_DIR)
