import statistics
from enum import Enum, auto
from typing import Optional, Union, Any

import math
import numpy as np

from metrics import computeAllManualRegressionMetrics, computeAllManualBinaryMetrics
from preprocessor import DatasetSplit
from utils_date import timestampToDateObj, timestampToDateStr, SAVE_DATE_FORMAT, getNextWorkDays, dateObjToTimestamp
from utils_misc import weightedAverage
from utils_random import randInt

_sequential_values_cache = {}
_r_sequential_values_cache = {}
_exp_sequential_values_cache = {}
_exp_r_sequential_values_cache = {}


def syncPredictionsAndLabelsSize(predictions: Union[list[Optional[float]], list[tuple]],
                                 labels: Union[list[Optional[float]], list[tuple]]) -> tuple:
    if type(predictions[0]) is tuple:
        _, predictions = list(zip(*predictions))
    if type(labels[0]) is tuple:
        _, labels = list(zip(*labels))
    predictions = list(filter(lambda x: x is not None, predictions))
    labels = list(filter(lambda x: x is not None, labels))
    future = predictions[len(labels):]
    predictions = predictions[:len(labels)]
    return predictions, labels, future


def matchValueWithDate(values: list[Optional[float]], source: DatasetSplit, index_tracker: dict, dates_mapper: dict,
                       date_str_out: bool = False, date_obj_out: bool = False) -> list[tuple]:
    matched = []
    future_data = []
    last_date = None
    for i, price in enumerate(values):
        if price is not None:
            index = index_tracker.get((source, i), None)
            if index is not None:
                date = dates_mapper[index]
                last_date = date
                if date_obj_out:
                    date = timestampToDateObj(date)
                elif date_str_out:
                    date = timestampToDateStr(date, output_format=SAVE_DATE_FORMAT)
                matched.append((date, price))
            else:
                future_data.append(price)
    if len(future_data) > 0:
        future_dates = getNextWorkDays(timestampToDateObj(last_date), len(future_data))
        for date, price in zip(future_dates, future_data):
            if not date_obj_out:
                date = dateObjToTimestamp(date)
                if date_str_out:
                    date = timestampToDateStr(date, output_format=SAVE_DATE_FORMAT)
            matched.append((date, price))
    return matched


def getFutureDates(future: list[Optional[float]], past_size: int, index_tracker: dict, dates_mapper: dict,
                   date_str_out: bool = False, date_obj_out: bool = False) -> list[Any]:
    if index_tracker.get((DatasetSplit.TEST, past_size), None) is not None:
        raise ValueError('Wrong past size value!')
    index = index_tracker.get((DatasetSplit.TEST, past_size - 1), None)
    last_date = dates_mapper[index]
    future_dates = getNextWorkDays(timestampToDateObj(last_date), len(future))
    dates = []
    for date in future_dates:
        if not date_obj_out:
            date = dateObjToTimestamp(date)
            if date_str_out:
                date = timestampToDateStr(date, output_format=SAVE_DATE_FORMAT)
        dates.append(date)
    return dates


def matchValueWithDateAndGetDates(values: list[Optional[float]], source: DatasetSplit, index_tracker: dict,
                                  dates_mapper: dict, get_future_dates: bool = False, date_str_out: bool = False,
                                  date_obj_out: bool = False) -> list[tuple]:
    dates = []
    missing_days = 0
    last_date = None
    for i, price in enumerate(values):
        if price is not None:
            index = index_tracker.get((source, i), None)
            if index is not None:
                date = dates_mapper[index]
                last_date = date
                if date_obj_out:
                    date = timestampToDateObj(date)
                elif date_str_out:
                    date = timestampToDateStr(date, output_format=SAVE_DATE_FORMAT)
                dates.append(date)
            else:
                missing_days += 1
    if get_future_dates and missing_days > 0:
        future_dates = getNextWorkDays(timestampToDateObj(last_date), missing_days)
        for date in future_dates:
            if not date_obj_out:
                date = dateObjToTimestamp(date)
                if date_str_out:
                    date = timestampToDateStr(date, output_format=SAVE_DATE_FORMAT)
            dates.append(date)
    return dates


def decodeWindowedLabels(labels: np.ndarray, backward_samples: int) -> list[Optional[float]]:
    parsed_labels = [None for _ in range(labels.shape[0] + backward_samples)]
    for i, window_predictions in enumerate(labels):
        parsed_labels[backward_samples + i] = window_predictions[0]
    parsed_labels += labels[-1][1:].tolist()
    return parsed_labels


def decodeWindowedPredictions(predictions: np.ndarray, backward_samples: int,
                              forward_samples: int) -> list[Optional[list]]:
    parsed_predictions = [[] for _ in range(predictions.shape[0] + forward_samples - 1 + backward_samples)]
    for i in range(backward_samples):
        parsed_predictions[i] = None
    for i, window_predictions in enumerate(predictions):
        for j, day_prediction in enumerate(window_predictions):
            parsed_predictions[backward_samples + i + j].append(day_prediction)
    return parsed_predictions


def mergeDecodedPredictions(dec_train: Optional[list[Optional[list]]], dec_val: Optional[list[Optional[list]]],
                            dec_test: Optional[list[Optional[list]]]) -> list[Optional[list]]:
    merged = []
    if dec_train is not None:
        merged += dec_train
    if dec_val is not None:
        merged += dec_val
    if dec_test is not None:
        merged += dec_test
    return merged


class AggregationMethod(Enum):
    FIRST = auto()
    LAST = auto()
    MEAN = auto()
    MEDIAN = auto()
    FIRST_LAST_MEAN = auto()
    FIRST_LAST_MEDIAN = auto()
    F_WEIGHTED_AVERAGE = auto()
    L_WEIGHTED_AVERAGE = auto()
    EXP_F_WEIGHTED_AVERAGE = auto()
    EXP_L_WEIGHTED_AVERAGE = auto()
    VOTING_MEAN = auto()
    VOTING_MEDIAN = auto()
    VOTING_F_WEIGHTED_AVERAGE = auto()
    VOTING_L_WEIGHTED_AVERAGE = auto()
    VOTING_EXP_F_WEIGHTED_AVERAGE = auto()
    VOTING_EXP_L_WEIGHTED_AVERAGE = auto()

    def __str__(self) -> str:
        return self.name.lower().replace('_', ' ')

    def toStrNoSpace(self):
        return AggregationMethod.strNoSpace(self)

    @staticmethod
    def strNoSpace(agg: Union[Enum, str]):
        if type(agg) is str:
            return agg.lower().replace(' ', '_')
        return agg.name.lower()

    @staticmethod
    def getAll():
        return list(map(lambda c: c, AggregationMethod))


def _aggFirst(predictions: Optional[list[Optional[list]]]) -> list[Optional[float]]:
    aggregated = []
    for values in predictions:
        if values is None:
            aggregated.append(values)
        else:
            aggregated.append(values[0])
    return aggregated


def _aggLast(predictions: Optional[list[Optional[list]]]) -> list[Optional[float]]:
    aggregated = []
    for values in predictions:
        if values is None:
            aggregated.append(values)
        else:
            aggregated.append(values[-1])
    return aggregated


def _aggMean(predictions: Optional[list[Optional[list]]]) -> list[Optional[float]]:
    aggregated = []
    for values in predictions:
        if values is None:
            aggregated.append(values)
        elif len(values) == 1:
            aggregated.append(values[0])
        else:
            aggregated.append(statistics.mean(values))
    return aggregated


def _aggMedian(predictions: Optional[list[Optional[list]]]) -> list[Optional[float]]:
    aggregated = []
    for values in predictions:
        if values is None:
            aggregated.append(values)
        elif len(values) == 1:
            aggregated.append(values[0])
        elif len(values) == 2:
            aggregated.append(statistics.mean(values))
        else:
            aggregated.append(statistics.median(values))
    return aggregated


def _aggFLMean(predictions: Optional[list[Optional[list]]]) -> list[Optional[float]]:
    aggregated = []
    for values in predictions:
        if values is None:
            aggregated.append(values)
        elif len(values) == 1:
            aggregated.append(values[0])
        else:
            aggregated.append(statistics.mean([values[0], values[-1]]))
    return aggregated


def _aggFLMedian(predictions: Optional[list[Optional[list]]]) -> list[Optional[float]]:
    aggregated = []
    for values in predictions:
        if values is None:
            aggregated.append(values)
        elif len(values) == 1:
            aggregated.append(values[0])
        elif len(values) == 2:
            aggregated.append(statistics.mean(values))
        else:
            aggregated.append(statistics.median([values[0], values[-1]]))
    return aggregated


def getSeqWeights(n: int) -> list[int]:
    global _sequential_values_cache
    if n not in _sequential_values_cache:
        _sequential_values_cache[n] = list(range(1, n + 1, 1))
    return _sequential_values_cache[n]


def _aggFWAverage(predictions: Optional[list[Optional[list]]]) -> list[Optional[float]]:
    aggregated = []
    for values in predictions:
        if values is None:
            aggregated.append(values)
        elif len(values) == 1:
            aggregated.append(values[0])
        else:
            aggregated.append(weightedAverage(values, getRSeqWeights(len(values))))
    return aggregated


def getRSeqWeights(n: int) -> list[int]:
    global _r_sequential_values_cache
    if n not in _r_sequential_values_cache:
        _r_sequential_values_cache[n] = list(range(n, 0, -1))
    return _r_sequential_values_cache[n]


def _aggLWAverage(predictions: Optional[list[Optional[list]]]) -> list[Optional[float]]:
    aggregated = []
    for values in predictions:
        if values is None:
            aggregated.append(values)
        elif len(values) == 1:
            aggregated.append(values[0])
        else:
            aggregated.append(weightedAverage(values, getSeqWeights(len(values))))
    return aggregated


def getExpSeqWeights(n: int) -> list[int]:
    global _exp_sequential_values_cache
    if n not in _exp_sequential_values_cache:
        _exp_sequential_values_cache[n] = [math.exp(i) for i in range(1, n + 1, 1)]
    return _exp_sequential_values_cache[n]


def _aggExpFWAverage(predictions: Optional[list[Optional[list]]]) -> list[Optional[float]]:
    aggregated = []
    for values in predictions:
        if values is None:
            aggregated.append(values)
        elif len(values) == 1:
            aggregated.append(values[0])
        else:
            aggregated.append(weightedAverage(values, getExpRSeqWeights(len(values))))
    return aggregated


def getExpRSeqWeights(n: int) -> list[int]:
    global _exp_r_sequential_values_cache
    if n not in _exp_r_sequential_values_cache:
        _exp_r_sequential_values_cache[n] = [math.exp(i) for i in range(n, 0, -1)]
    return _exp_r_sequential_values_cache[n]


def _aggExpLWAverage(predictions: Optional[list[Optional[list]]]) -> list[Optional[float]]:
    aggregated = []
    for values in predictions:
        if values is None:
            aggregated.append(values)
        elif len(values) == 1:
            aggregated.append(values[0])
        else:
            aggregated.append(weightedAverage(values, getExpSeqWeights(len(values))))
    return aggregated


def _aggVotingMean(predictions: Optional[list[Optional[list]]]) -> list[Optional[float]]:
    aggregated = []
    last_value = None
    for values in predictions:
        if values is None:
            aggregated.append(values)
        else:
            if len(values) == 1:
                aggregated.append(values[0])
            elif len(values) == 2:
                aggregated.append(statistics.mean(values))
            else:
                up_values = []
                down_values = []
                up_down_values = [up_values, down_values]
                for value in values:
                    if value > last_value:
                        up_values.append(value)
                    elif value < last_value:
                        down_values.append(value)
                    else:
                        up_down_values[randInt(1)].append(value)
                if len(up_values) > len(down_values):
                    voted = up_values
                elif len(up_values) < len(down_values):
                    voted = down_values
                else:
                    voted = up_down_values[randInt(1)]
                aggregated.append(statistics.mean(voted))
            last_value = aggregated[-1]
    return aggregated


def _aggVotingMedian(predictions: Optional[list[Optional[list]]]) -> list[Optional[float]]:
    aggregated = []
    last_value = None
    for values in predictions:
        if values is None:
            aggregated.append(values)
        else:
            if len(values) == 1:
                aggregated.append(values[0])
            elif len(values) == 2:
                aggregated.append(statistics.mean(values))
            else:
                up_values = []
                down_values = []
                up_down_values = [up_values, down_values]
                for value in values:
                    if value > last_value:
                        up_values.append(value)
                    elif value < last_value:
                        down_values.append(value)
                    else:
                        up_down_values[randInt(1)].append(value)
                if len(up_values) > len(down_values):
                    voted = up_values
                elif len(up_values) < len(down_values):
                    voted = down_values
                else:
                    voted = up_down_values[randInt(1)]
                aggregated.append(statistics.median(voted))
            last_value = aggregated[-1]
    return aggregated


def _aggVotingFWAverage(predictions: Optional[list[Optional[list]]]) -> list[Optional[float]]:
    aggregated = []
    last_value = None
    for values in predictions:
        if values is None:
            aggregated.append(values)
        else:
            if len(values) == 1:
                aggregated.append(values[0])
            elif len(values) == 2:
                aggregated.append(statistics.mean(values))
            else:
                up_values = []
                down_values = []
                up_down_values = [up_values, down_values]
                for value in values:
                    if value > last_value:
                        up_values.append(value)
                    elif value < last_value:
                        down_values.append(value)
                    else:
                        up_down_values[randInt(1)].append(value)
                if len(up_values) > len(down_values):
                    voted = up_values
                elif len(up_values) < len(down_values):
                    voted = down_values
                else:
                    voted = up_down_values[randInt(1)]
                aggregated.append(weightedAverage(voted, getRSeqWeights(len(voted))))
            last_value = aggregated[-1]
    return aggregated


def _aggVotingLWAverage(predictions: Optional[list[Optional[list]]]) -> list[Optional[float]]:
    aggregated = []
    last_value = None
    for values in predictions:
        if values is None:
            aggregated.append(values)
        else:
            if len(values) == 1:
                aggregated.append(values[0])
            elif len(values) == 2:
                aggregated.append(statistics.mean(values))
            else:
                up_values = []
                down_values = []
                up_down_values = [up_values, down_values]
                for value in values:
                    if value > last_value:
                        up_values.append(value)
                    elif value < last_value:
                        down_values.append(value)
                    else:
                        up_down_values[randInt(1)].append(value)
                if len(up_values) > len(down_values):
                    voted = up_values
                elif len(up_values) < len(down_values):
                    voted = down_values
                else:
                    voted = up_down_values[randInt(1)]
                aggregated.append(weightedAverage(voted, getSeqWeights(len(voted))))
            last_value = aggregated[-1]
    return aggregated


def _aggVotingExpFWAverage(predictions: Optional[list[Optional[list]]]) -> list[Optional[float]]:
    aggregated = []
    last_value = None
    for values in predictions:
        if values is None:
            aggregated.append(values)
        else:
            if len(values) == 1:
                aggregated.append(values[0])
            elif len(values) == 2:
                aggregated.append(statistics.mean(values))
            else:
                up_values = []
                down_values = []
                up_down_values = [up_values, down_values]
                for value in values:
                    if value > last_value:
                        up_values.append(value)
                    elif value < last_value:
                        down_values.append(value)
                    else:
                        up_down_values[randInt(1)].append(value)
                if len(up_values) > len(down_values):
                    voted = up_values
                elif len(up_values) < len(down_values):
                    voted = down_values
                else:
                    voted = up_down_values[randInt(1)]
                aggregated.append(weightedAverage(voted, getExpRSeqWeights(len(voted))))
            last_value = aggregated[-1]
    return aggregated


def _aggVotingExpLWAverage(predictions: Optional[list[Optional[list]]]) -> list[Optional[float]]:
    aggregated = []
    last_value = None
    for values in predictions:
        if values is None:
            aggregated.append(values)
        else:
            if len(values) == 1:
                aggregated.append(values[0])
            elif len(values) == 2:
                aggregated.append(statistics.mean(values))
            else:
                up_values = []
                down_values = []
                up_down_values = [up_values, down_values]
                for value in values:
                    if value > last_value:
                        up_values.append(value)
                    elif value < last_value:
                        down_values.append(value)
                    else:
                        up_down_values[randInt(1)].append(value)
                if len(up_values) > len(down_values):
                    voted = up_values
                elif len(up_values) < len(down_values):
                    voted = down_values
                else:
                    voted = up_down_values[randInt(1)]
                aggregated.append(weightedAverage(voted, getExpSeqWeights(len(voted))))
            last_value = aggregated[-1]
    return aggregated


def aggregateDecodedPredictions(predictions: Optional[list[Optional[list]]],
                                method: AggregationMethod) -> list[Optional[float]]:
    if method == AggregationMethod.FIRST:
        return _aggFirst(predictions)
    elif method == AggregationMethod.LAST:
        return _aggLast(predictions)
    elif method == AggregationMethod.MEAN:
        return _aggMean(predictions)
    elif method == AggregationMethod.MEDIAN:
        return _aggMedian(predictions)
    elif method == AggregationMethod.FIRST_LAST_MEAN:
        return _aggFLMean(predictions)
    elif method == AggregationMethod.FIRST_LAST_MEDIAN:
        return _aggFLMedian(predictions)
    elif method == AggregationMethod.F_WEIGHTED_AVERAGE:
        return _aggFWAverage(predictions)
    elif method == AggregationMethod.L_WEIGHTED_AVERAGE:
        return _aggLWAverage(predictions)
    elif method == AggregationMethod.EXP_F_WEIGHTED_AVERAGE:
        return _aggExpFWAverage(predictions)
    elif method == AggregationMethod.EXP_L_WEIGHTED_AVERAGE:
        return _aggExpLWAverage(predictions)
    elif method == AggregationMethod.VOTING_MEAN:
        return _aggVotingMean(predictions)
    elif method == AggregationMethod.VOTING_MEDIAN:
        return _aggVotingMedian(predictions)
    elif method == AggregationMethod.VOTING_F_WEIGHTED_AVERAGE:
        return _aggVotingFWAverage(predictions)
    elif method == AggregationMethod.VOTING_L_WEIGHTED_AVERAGE:
        return _aggVotingLWAverage(predictions)
    elif method == AggregationMethod.VOTING_EXP_F_WEIGHTED_AVERAGE:
        return _aggVotingExpFWAverage(predictions)
    elif method == AggregationMethod.VOTING_EXP_L_WEIGHTED_AVERAGE:
        return _aggVotingExpLWAverage(predictions)
    else:
        raise ValueError(f'Unknown method {method}')


def aggregateDecodedPredictionsAllMethods(predictions: Optional[list[Optional[list]]]) -> dict:
    out = {}
    for method in AggregationMethod.getAll():
        out[str(method)] = aggregateDecodedPredictions(predictions, method)
    return out


def aggregateAllDecodedPredictionsAllMethods(predictions: dict) -> dict:
    out = {}
    for d_type, data in predictions.items():
        decoded = {}
        for method in AggregationMethod.getAll():
            decoded[str(method)] = aggregateDecodedPredictions(data, method)
        out[d_type] = decoded
    return out


def castRegressionToDelta(array: list[Union[np.float32, float]],
                          previous_value: Union[np.float32, float]) -> list[Union[np.float32, float]]:
    deltas = []
    for i in range(len(array)):
        deltas.append(array[i] - previous_value)
        previous_value = array[i]
    return deltas


def castRegressionToBinary(array: list[Union[np.float32, float]],
                           previous_value: Union[np.float32, float]) -> list[int]:
    deltas = castRegressionToDelta(array, previous_value)
    return [1 if el > 0 else 0 for el in deltas]


def computeMetricsAndGetValues(predictions: Union[dict, Optional[list[Optional[list]]]], labels: dict,
                               prev_labels: dict, index_tracker: dict, dates_mapper: dict) -> dict:
    all_metrics_and_values = {}
    if type(predictions) is not dict:
        predictions = {'predictions': predictions}
    all_predictions_all_agg = aggregateAllDecodedPredictionsAllMethods(predictions)
    for d_type, predictions_all_agg in all_predictions_all_agg.items():
        metrics_and_values = {}
        label = labels[d_type]
        dates = matchValueWithDateAndGetDates(list(predictions_all_agg.values())[0], d_type, index_tracker,
                                              dates_mapper, get_future_dates=False, date_obj_out=True)
        binary_labels = None
        metrics_and_values['dates'] = dates
        metrics_and_values['predictions'] = {}
        for agg_method, agg_data in predictions_all_agg.items():
            predictions_synced, labels_synced, future = syncPredictionsAndLabelsSize(agg_data, label)
            if 'label_vals' not in metrics_and_values:
                metrics_and_values['label_vals'] = labels_synced
                binary_labels = castRegressionToBinary(labels_synced, prev_labels[d_type])
            binary_predictions = castRegressionToBinary(predictions_synced, prev_labels[d_type])
            # label=str(d_type).replace(' ', '_')
            r_metrics = computeAllManualRegressionMetrics(predictions_synced, labels_synced)
            b_metrics = computeAllManualBinaryMetrics(binary_predictions, binary_labels)
            metrics_and_values['predictions'][agg_method] = {
                'vals': predictions_synced,
                'metrics': {
                    'regression': r_metrics,
                    'binary': b_metrics
                }
            }
            if len(future) > 0:
                metrics_and_values['predictions'][agg_method]['future'] = {
                    'vals': future,
                    'dates': getFutureDates(future, len(label), index_tracker, dates_mapper, date_obj_out=True)
                }
        all_metrics_and_values[d_type] = metrics_and_values
    # if len(all_metrics_and_values) == 1:
    #     return list(all_metrics_and_values.values())[0]
    # else:
    return all_metrics_and_values
