import copy as cp
import math
from enum import Enum, auto
from typing import Union, Optional

import numpy as np
from sklearn.utils import shuffle

from hyperparameters import Hyperparameters
from logger import info, verbose
from prophet_filepaths import getTickerFromFilename
from transformer import DATE_COLUMN, PREDICT_COLUMN, getTickerScalerFilepath
from utils_date import dateStrToTimestamp, SAVE_DATE_FORMAT
from utils_fs import getBasename
from utils_misc import _add_two_way_dict, _add_four_way_dict
from utils_persistance import loadDataframe


def filterBackwardsRollingWindow(windows: list, window: int, exclusive: bool = False) -> tuple[list, list]:
    if not exclusive:
        window -= 1
    past_useless_data = windows[:window]
    useful_train_data = windows[window:]
    return useful_train_data, past_useless_data


def filterForwardRollingWindow(windows: list, window: int) -> tuple[list, list]:
    useful_train_data = windows[:-window]
    future_useless_data = windows[-window:]
    return useful_train_data, future_useless_data


def filterDoubleRollingWindow(backwards_windows: Union[list, np.ndarray], backward_window: int,
                              forward_windows: Union[list, np.ndarray], forward_window: int,
                              keep_future_features: bool = False) -> tuple[tuple, tuple]:
    is_np = type(backwards_windows) is np.ndarray or type(forward_windows) is np.ndarray
    useful_train_back_data, past_useless_data = filterBackwardsRollingWindow(backwards_windows, backward_window)
    useful_train_fut_data, future_useless_data = filterForwardRollingWindow(forward_windows, forward_window)

    diff = backward_window - forward_window
    past_idx = backward_window - diff
    future_idx = forward_window - 1 + diff

    if not keep_future_features:
        if is_np:
            past_useless_data = np.concatenate((past_useless_data, useful_train_back_data[-past_idx:]))
        else:
            past_useless_data += useful_train_back_data[-past_idx:]
    if is_np:
        future_useless_data = np.concatenate(
            (future_useless_data, useful_train_fut_data[:future_idx], future_useless_data))
    else:
        future_useless_data = useful_train_fut_data[:future_idx] + future_useless_data

    if not keep_future_features:
        useful_train_back_data = useful_train_back_data[:-past_idx]
    useful_train_fut_data = useful_train_fut_data[future_idx:]

    if is_np:
        useful_train_back_data = np.reshape(np.vstack(useful_train_back_data),
                                            ([useful_train_back_data.shape[0]] + list(useful_train_back_data[0].shape)))
        useful_train_fut_data = np.reshape(np.vstack(useful_train_fut_data),
                                           ([useful_train_fut_data.shape[0]] + list(useful_train_fut_data[0].shape)))
    useful_train_data = (useful_train_back_data, useful_train_fut_data)
    useless_data = (past_useless_data, future_useless_data)
    return useful_train_data, useless_data


def backwardsRollingWindowInc(array: Union[list, np.ndarray], window: int) -> Union[list, np.ndarray]:
    # returns past rolling window inclusively
    # return series.rolling(window=window)
    is_np = type(array) is np.ndarray
    windows = []
    for i in range(len(array)):
        i_plus_1 = i + 1
        if i_plus_1 < window:
            windows.append(array[i])  # base case
        else:
            this_window = array[i_plus_1 - window: i_plus_1][::-1]
            windows.append(this_window)
    return np.array(windows, dtype='object') if is_np else windows


def forwardRollingWindowInc(array: Union[list, np.ndarray], window: int) -> Union[list, np.ndarray]:
    # returns future rolling window inclusively
    # return series.rolling(window=window).shift(-(window - 1))
    is_np = type(array) is np.ndarray
    windows = []
    for i in range(len(array)):
        if i >= len(array) - window:
            windows.append(array[i])  # base case
        else:
            this_window = array[i: i + window]
            windows.append(this_window)
    return np.array(windows, dtype='object') if is_np else windows


def backwardsRollingWindowExc(array: Union[list, np.ndarray], window: int) -> Union[list, np.ndarray]:
    # returns past rolling window exclusively
    # return series.rolling(window=window).shift().bfill()
    is_np = type(array) is np.ndarray
    windows = [None]
    for i in range(1, len(array)):
        if i < window:
            windows.append(array[i - 1])  # base case
        else:
            this_window = array[i - window: i][::-1]
            windows.append(this_window)
    return np.array(windows, dtype='object') if is_np else windows


def forwardRollingWindowExc(array: Union[list, np.ndarray], window: int) -> Union[list, np.ndarray]:
    # returns future rolling window exclusively
    # return series.rolling(window=window).shift().bfill().shift(-(window - 1))
    is_np = type(array) is np.ndarray
    windows = []
    for i in range(len(array) - 1):
        if i >= len(array) - window:
            windows.append(array[i + 1])  # base case
        else:
            this_window = array[i + 1: i + 1 + window]
            windows.append(this_window)
    windows.append(None)
    return np.array(windows, dtype='object') if is_np else windows


class DatasetSplit(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()

    def __str__(self) -> str:
        return self.name.lower()

    @staticmethod
    def getAll():
        return list(map(lambda c: c, DatasetSplit))


def trainTestSplitRatio(features: list[np.ndarray], labels: list[np.ndarray], index_tracker: dict,
                        train_ratio: Union[float, int], test_type: DatasetSplit = DatasetSplit.TEST,
                        add_train_keys_on_tracker: bool = True) -> tuple[tuple[list, list], tuple[list, list]]:
    dataset_size = len(labels)
    last_train_idx = math.floor(dataset_size * train_ratio)
    features_train, features_test = features[:last_train_idx], features[last_train_idx:]
    labels_train, labels_test = labels[:last_train_idx], labels[last_train_idx:]
    for i in range(dataset_size):
        if i < last_train_idx:
            if add_train_keys_on_tracker:
                _add_two_way_dict(index_tracker, i, (DatasetSplit.TRAIN, i))
        else:
            _add_two_way_dict(index_tracker, i, (test_type, i - last_train_idx))
    splitted_data = (
        (features_train, features_test,),
        (labels_train, labels_test,),
    )
    return splitted_data


class ProcessedDataset:
    pass  # Just to hint its return type in the true class


class ProcessedDataset(object):
    def __init__(self, features_train: Optional[np.ndarray] = None, labels_train: Optional[np.ndarray] = None,
                 features_val: Optional[np.ndarray] = None, labels_val: Optional[np.ndarray] = None,
                 features_test: Optional[np.ndarray] = None, labels_test: Optional[np.ndarray] = None,
                 prev_label_train: Optional[np.float32] = None, prev_label_val: Optional[np.float32] = None,
                 prev_label_test: Optional[np.float32] = None, index_tracker: Optional[dict] = None,
                 dates_mapper: Optional[dict] = None, scaler_path: Optional[str] = None):
        self.features_train = features_train
        self.labels_train = labels_train
        self.features_val = features_val
        self.labels_val = labels_val
        self.features_test = features_test
        self.labels_test = labels_test
        self.prev_label_train = prev_label_train
        self.prev_label_val = prev_label_val
        self.prev_label_test = prev_label_test
        self.index_tracker = index_tracker
        self.dates_mapper = dates_mapper
        self.scaler_path = scaler_path
        self.ticker = None
        if scaler_path is not None:
            self.ticker = getTickerFromFilename(self.scaler_path)
        self.encoded = False

    def hasTrain(self) -> bool:
        return self.features_train is not None and self.labels_train is not None and self.features_train.shape[
            0] > 0 and self.labels_train.shape[0] > 0

    def hasVal(self) -> bool:
        return self.features_val is not None and self.labels_val is not None and self.features_val.shape[0] > 0 and \
               self.labels_val.shape[0] > 0

    def hasTest(self) -> bool:
        return self.features_test is not None and self.labels_test is not None and self.features_test.shape[0] > 0 and \
               self.labels_test.shape[0] > 0

    def hashEval(self) -> bool:
        return self.features_test is not None and self.features_test.shape[0] > 0

    def getFeaturesDict(self) -> dict:
        data = {}
        if self.hasTrain():
            data[DatasetSplit.TRAIN] = self.features_train
        if self.hasVal():
            data[DatasetSplit.VALIDATION] = self.features_val
        if self.hashEval():
            data[DatasetSplit.TEST] = self.features_test
        return data

    def getFeaturesAndLabelsDict(self) -> dict:
        data = {}
        if self.hasTrain():
            data[DatasetSplit.TRAIN] = (self.features_train, self.labels_train)
        if self.hasVal():
            data[DatasetSplit.VALIDATION] = (self.features_val, self.labels_val)
        if self.hashEval():
            data[DatasetSplit.TEST] = (self.features_test[:len(self.labels_test)], self.labels_test)
        return data

    def getPreviousLabelsDict(self) -> dict:
        data = {}
        if self.prev_label_train is not None:
            data[DatasetSplit.TRAIN] = self.prev_label_train
        if self.prev_label_val is not None:
            data[DatasetSplit.VALIDATION] = self.prev_label_val
        if self.prev_label_test is not None:
            data[DatasetSplit.TEST] = self.prev_label_test
        return data

    def copy(self, copy_index_and_tracker: bool = True) -> ProcessedDataset:
        features_train = cp.deepcopy(self.features_train)
        labels_train = cp.deepcopy(self.labels_train)
        features_val = cp.deepcopy(self.features_val)
        labels_val = cp.deepcopy(self.labels_val)
        features_test = cp.deepcopy(self.features_test)
        labels_test = cp.deepcopy(self.labels_test)
        if copy_index_and_tracker:
            index_tracker = cp.deepcopy(self.index_tracker)
            dates_mapper = cp.deepcopy(self.dates_mapper)
        else:
            index_tracker = self.index_tracker
            dates_mapper = self.dates_mapper
        that = ProcessedDataset(features_train=features_train, labels_train=labels_train, features_val=features_val,
                                labels_val=labels_val,
                                features_test=features_test, labels_test=labels_test,
                                prev_label_train=self.prev_label_train, prev_label_val=self.prev_label_val,
                                prev_label_test=self.prev_label_test, index_tracker=index_tracker,
                                dates_mapper=dates_mapper, scaler_path=self.scaler_path)
        return that

    def encode(self, configs: Optional[Hyperparameters] = None, backward_samples: Optional[int] = None,
               forward_samples: Optional[int] = None, copy: bool = False) -> ProcessedDataset:

        if configs is None and backward_samples is None and forward_samples is None:
            raise AttributeError('Missing arguments')
        if backward_samples is None:
            backward_samples = configs.network.backward_samples
        if forward_samples is None:
            forward_samples = configs.network.forward_samples

        to_enc = self.copy() if copy else self
        if to_enc.hasTrain():
            if configs.network.shuffle:
                to_enc.features_train, to_enc.labels_train = shuffle(to_enc.features_train, to_enc.labels_train)
            to_enc.features_train, to_enc.labels_train = encodeRollingWindowsNumpySafe(to_enc.features_train,
                                                                                       to_enc.labels_train,
                                                                                       backward_samples,
                                                                                       forward_samples)
            to_enc.labels_train = np.squeeze(to_enc.labels_train)
        if to_enc.hasVal():
            to_enc.features_val, to_enc.labels_val = encodeRollingWindowsNumpySafe(to_enc.features_val,
                                                                                   to_enc.labels_val,
                                                                                   backward_samples,
                                                                                   forward_samples)
            to_enc.labels_val = np.squeeze(to_enc.labels_val)
        if to_enc.hasTest():
            to_enc.features_test, to_enc.labels_test = encodeRollingWindowsNumpySafe(to_enc.features_test,
                                                                                     to_enc.labels_test,
                                                                                     backward_samples,
                                                                                     forward_samples, True)
            to_enc.labels_test = np.squeeze(to_enc.labels_test)
        to_enc.encoded = True
        return to_enc


def _verbose_arrange(finished: bool) -> None:
    if finished:
        verbose('Arranged data for LSTM and casted to numpy!')
    else:
        verbose('Arranging data for LSTM and casting to numpy...')


def _verbose_finished_split(split_validation: bool) -> None:
    verbose(f'Splitted data into train,{"val, " if split_validation else " "}and test and created tracker!')


def preprocess(filepath: str, encode_rolling_window: bool = True, load_test: bool = True,
               configs: Optional[Hyperparameters] = None) -> ProcessedDataset:
    if configs is None:
        configs = Hyperparameters.getDefault()
    filename = getBasename(filepath)
    info(f'Pre-processing dataset {filename}...')
    df = loadDataframe(filepath)
    feature_columns = df.columns.difference([DATE_COLUMN, PREDICT_COLUMN])

    index_tracker = {}
    dates_mapper = {}
    features = []
    labels = []
    verbose('Preparing data and creating mapper...')
    for i, (idx, row) in enumerate(df.iterrows()):
        date_str = row[DATE_COLUMN]
        date_ts = dateStrToTimestamp(date_str, input_format=SAVE_DATE_FORMAT)
        date_ts_int = int(date_ts)
        x = row[feature_columns].to_numpy()
        y = np.full(shape=1, fill_value=row[PREDICT_COLUMN])
        _add_four_way_dict(dates_mapper, date_str, date_ts, date_ts_int, i)
        features.append(x)
        labels.append(y)
    verbose('Prepared data and created mapper!')

    split_validation = configs.dataset.validation_ratio > 0
    verbose(f'Splitting data into train,{"val, " if split_validation else " "}and test and creating tracker...')
    old_labels = labels
    features, labels = trainTestSplitRatio(features, labels, index_tracker, configs.dataset.train_ratio,
                                           add_train_keys_on_tracker=not split_validation)
    if not split_validation:
        _verbose_finished_split(split_validation)

        features_train = features[0]
        labels_train = labels[0]
        features_test = features[1]
        labels_test = labels[1]
        if not load_test:
            features_test = labels_test = []

        features_train = np.array(features_train).astype(np.float32)
        labels_train = np.array(labels_train).astype(np.float32)
        features_val = np.array([]).astype(np.float32)
        labels_val = np.array([]).astype(np.float32)
        features_test = np.array(features_test).astype(np.float32)
        labels_test = np.array(labels_test).astype(np.float32)

        prev_label_train = old_labels[configs.network.backward_samples - 1][0]
        prev_label_val = None
        if load_test:
            p_test_index = features_train.shape[
                               0] + configs.network.backward_samples * 2 + configs.network.forward_samples - 1
            prev_label_test = old_labels[p_test_index][0]
        else:
            prev_label_test = None
    else:  # has validation
        features_train_val, labels_train_val = trainTestSplitRatio(features[0], labels[0], index_tracker,
                                                                   1 - configs.dataset.validation_ratio,
                                                                   DatasetSplit.VALIDATION)
        _verbose_finished_split(split_validation)

        features_train = features_train_val[0]
        labels_train = labels_train_val[0]
        features_val = features_train_val[1]
        labels_val = labels_train_val[1]
        features_test = features[1]
        labels_test = labels[1]
        if not load_test:
            features_test = labels_test = []

        features_train = np.array(features_train).astype(np.float32)
        labels_train = np.array(labels_train).astype(np.float32)
        features_val = np.array(features_val).astype(np.float32)
        labels_val = np.array(labels_val).astype(np.float32)
        features_test = np.array(features_test).astype(np.float32)
        labels_test = np.array(labels_test).astype(np.float32)

        prev_label_train = old_labels[configs.network.backward_samples - 1][0]
        p_val_index = features_train.shape[
                          0] + configs.network.backward_samples * 2 + configs.network.forward_samples - 1
        prev_label_val = old_labels[p_val_index][0]
        if load_test:
            p_test_index = p_val_index + features_val.shape[
                0] + configs.network.backward_samples + configs.network.forward_samples
            prev_label_test = old_labels[p_test_index][0]
        else:
            prev_label_test = None

    processed_data = ProcessedDataset(features_train=features_train, labels_train=labels_train,
                                      features_val=features_val, labels_val=labels_val,
                                      features_test=features_test, labels_test=labels_test,
                                      prev_label_train=prev_label_train, prev_label_val=prev_label_val,
                                      prev_label_test=prev_label_test, index_tracker=index_tracker,
                                      dates_mapper=dates_mapper, scaler_path=getTickerScalerFilepath(filename))
    if encode_rolling_window:
        _verbose_arrange(False)
        processed_data.encode(configs)
        _verbose_arrange(True)

    info(f'Pre-processed dataset {filename}!')
    return processed_data


def encodeRollingWindows(features: Union[list[np.ndarray], np.ndarray], labels: Union[list[np.ndarray], np.ndarray],
                         backward_samples: int, forward_samples: int,
                         keep_future_features: bool = False) -> Union[tuple[list, list], tuple[np.ndarray, np.ndarray]]:
    # encode in: (samples, features)
    # network in: (samples, backward_samples, features)
    # network out: (samples, forward_samples)
    features = backwardsRollingWindowInc(features, backward_samples)
    labels = forwardRollingWindowExc(labels, forward_samples)
    useful_train_data, _ = filterDoubleRollingWindow(features, backward_samples, labels, forward_samples,
                                                     keep_future_features)
    return useful_train_data


def encodeRollingWindowsNumpySafe(features: Union[list[np.ndarray], np.ndarray],
                                  labels: Union[list[np.ndarray], np.ndarray],
                                  backward_samples: int, forward_samples: int,
                                  keep_future_features: bool = False) -> tuple:
    # created because backwardsRollingWindowInc and forwardRollingWindowExc were failing when the window where equal to
    # amount of features, this one is also faster
    is_np = type(features) is np.ndarray
    # backwardsRollingWindowInc
    useful_train_back_data = []
    past_useless_data = []
    for i in range(len(features)):
        i_plus_1 = i + 1
        if i_plus_1 < backward_samples:
            past_useless_data.append(features[i])  # base case
        else:
            this_window = features[i_plus_1 - backward_samples: i_plus_1][::-1]
            useful_train_back_data.append(this_window)
    if is_np:
        useful_train_back_data = np.array(useful_train_back_data)
        past_useless_data = np.array(past_useless_data)

    # forwardRollingWindowExc
    useful_train_fut_data = []
    for i in range(len(labels) - 1):
        if i >= len(labels) - forward_samples:
            pass  # base case
        else:
            this_window = labels[i + 1: i + 1 + forward_samples]
            useful_train_fut_data.append(this_window)
    if is_np:
        useful_train_fut_data = np.array(useful_train_fut_data)

    # filterDoubleRollingWindow
    diff = backward_samples - forward_samples
    past_idx = backward_samples - diff
    future_idx = forward_samples - 1 + diff

    if not keep_future_features:
        if not is_np:
            past_useless_data += useful_train_back_data[-past_idx:]

    if not keep_future_features:
        useful_train_back_data = useful_train_back_data[:-past_idx]
    useful_train_fut_data = useful_train_fut_data[future_idx:]

    if is_np:
        useful_train_back_data = np.reshape(np.vstack(useful_train_back_data),
                                            ([useful_train_back_data.shape[0]] + list(useful_train_back_data[0].shape)))
        useful_train_fut_data = np.reshape(np.vstack(useful_train_fut_data),
                                           ([useful_train_fut_data.shape[0]] + list(useful_train_fut_data[0].shape)))
    return useful_train_back_data, useful_train_fut_data
