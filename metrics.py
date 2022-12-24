import os
import warnings
from enum import Enum, auto
from typing import Union, Callable, Optional

import math
import numpy as np
from sklearn import metrics as sk_metrics
from sklearn.exceptions import UndefinedMetricWarning

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # DISABLE TENSORFLOW WARNING
import tensorflow as tf

EPSILON = 1e-07
DROP_NAN_BEFORE_COMPUTE_METRICS = True


class Metric(Enum):
    RAW_LOSS = auto()
    ACCURACY = auto()
    MSE = auto()
    MAE = auto()
    COSINE_SIM = auto()
    F1 = auto()
    PRECISION = auto()
    RECALL = auto()
    RMSE = auto()
    R2 = auto()
    MAPE = auto()

    def toKerasName(self) -> str:
        if self == Metric.RAW_LOSS:
            return 'loss'
        elif self == Metric.F1:
            return 'f1_score'
        elif self == Metric.RECALL:
            return 'recall'
        elif self == Metric.ACCURACY:
            return 'accuracy'
        elif self == Metric.PRECISION:
            return 'precision'
        elif self == Metric.MSE:
            return 'mean_squared_error'
        elif self == Metric.MAE:
            return 'mean_absolute_error'
        elif self == Metric.COSINE_SIM:
            return 'cosine_similarity'
        elif self == Metric.RMSE:
            return 'root_mean_squared_error'
        elif self == Metric.R2:
            return 'R2'
        elif self == Metric.MAPE:
            return 'mean_absolute_percentage_error'
        raise ValueError('Strange error, invalid enum')

    @staticmethod
    def getAll() -> list[Enum]:
        return list(map(lambda c: c, Metric))

    @staticmethod
    def getUpperBound() -> int:
        return max([x.value for x in Metric.getAll()])


def createR2TfMetric() -> Callable:
    def r_squared(y_true, y_pred):
        # SS_res = K.sum(K.square(y_true - y_pred))
        # SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        # return (1 - SS_res / (SS_tot + K.epsilon()))
        residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
        total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
        r2 = tf.subtract(1.0, tf.divide(residual, total))
        return r2

    r_squared.__name__ = 'R2'
    return r_squared


def getAllCustomMetrics() -> dict:
    custom = [createR2TfMetric()]
    custom_dict = {}
    for c in custom:
        custom_dict[c.__name__] = c
    return custom_dict


def getRegressionMetrics(get_names_only: bool = False) -> list[Union[str, Callable]]:
    metrics = [
        Metric.R2,
        Metric.MSE,
        Metric.MAE,
        Metric.ACCURACY,
        Metric.COSINE_SIM,
        # Metric.RMSE,
        Metric.MAPE
    ]
    for i, metric in enumerate(metrics):
        if not get_names_only and metric == Metric.R2:
            metrics[i] = createR2TfMetric()
        else:
            metrics[i] = metric.toKerasName()
    return metrics


def manualMeanAbsoluteError(predictions: list[Optional[float]], labels: list[Optional[float]]) -> float:
    if len(predictions) != len(labels):
        raise AttributeError('Predictions and labels must have the same size')
    mae_sum = 0
    for predict, label in zip(predictions, labels):
        error = label - predict
        mae_sum += abs(error)
    mae = mae_sum / len(labels)
    return mae


def manualMeanAbsolutePercentageError(predictions: list[Optional[float]], labels: list[Optional[float]]) -> float:
    if len(predictions) != len(labels):
        raise AttributeError('Predictions and labels must have the same size')
    mape_sum = 0
    for predict, label in zip(predictions, labels):
        error = label - predict
        mape_sum += abs(error) / max(EPSILON, abs(label))
    mape = mape_sum / len(labels)
    return mape


def manualMeanSquaredError(predictions: list[Optional[float]], labels: list[Optional[float]]) -> float:
    if len(predictions) != len(labels):
        raise AttributeError('Predictions and labels must have the same size')
    mse_sum = 0
    for predict, label in zip(predictions, labels):
        error = label - predict
        mse_sum += error ** 2
    mse = mse_sum / len(labels)
    return mse


def manualRootMeanSquaredError(predictions: list[Optional[float]], labels: list[Optional[float]]) -> float:
    mse = manualMeanSquaredError(predictions, labels)
    rmse = math.sqrt(mse)
    return rmse


def manualR2(predictions: list[Optional[float]], labels: list[Optional[float]], clip: bool = True) -> float:
    if len(predictions) != len(labels):
        raise AttributeError('Predictions and labels must have the same size')
    rss = 0
    label_mean = 0
    for predict, label in zip(predictions, labels):
        error = label - predict
        rss += error ** 2
        label_mean += label
    label_mean = label_mean / len(labels)
    tss = 0
    for label in labels:
        distance = label - label_mean
        tss += distance ** 2
    r2 = 1 - (rss / max(EPSILON, tss))
    if clip and r2 < 0:
        r2 = 0
    return r2


def manualCosineSimilarity(predictions: list[Optional[float]], labels: list[Optional[float]],
                           clip: bool = True) -> float:
    if len(predictions) != len(labels):
        raise AttributeError('Predictions and labels must have the same size')
    dot_product = 0
    norm_predictions = 0
    norm_labels = 0
    for predict, label in zip(predictions, labels):
        dot_product += predict * label
        norm_predictions += predict ** 2
        norm_labels += label ** 2
    norm_predictions = math.sqrt(norm_predictions)
    norm_labels = math.sqrt(norm_labels)
    cos_sim = dot_product / max(EPSILON, (norm_predictions * norm_labels))
    if clip and cos_sim < 0:
        cos_sim = 0
    return cos_sim


def dropNanValuesSimultaneously(array_a: Union[np.ndarray, list],
                                array_b: Union[np.ndarray, list]) -> tuple[np.ndarray, np.ndarray]:
    if type(array_a) is list:
        array_a = np.array(array_a)
    if type(array_b) is list:
        array_b = np.array(array_b)
    nan_from_a = list(np.argwhere(np.isnan(array_a)).squeeze())
    nan_from_b = list(np.argwhere(np.isnan(array_b)).squeeze())
    nan_from_both = list(set(nan_from_a + nan_from_b))
    return np.delete(array_a, nan_from_both), np.delete(array_b, nan_from_both)


def computeAllManualRegressionMetrics(predictions: list[Optional[float]], labels: list[Optional[float]],
                                      label: Optional[str] = None, prefix: Optional[str] = None) -> dict:
    if DROP_NAN_BEFORE_COMPUTE_METRICS:
        predictions, labels = dropNanValuesSimultaneously(predictions, labels)
    label = f'_{label}' if label is not None else ''
    prefix = f'_{prefix}' if prefix is not None else ''
    metrics = {prefix + 'mae' + label: manualMeanAbsoluteError(predictions, labels),
               prefix + 'mape' + label: manualMeanAbsolutePercentageError(predictions, labels),
               prefix + 'mse' + label: manualMeanSquaredError(predictions, labels),
               prefix + 'rmse' + label: manualRootMeanSquaredError(predictions, labels),
               prefix + 'r2' + label: manualR2(predictions, labels),
               prefix + 'cos_sim' + label: manualCosineSimilarity(predictions, labels)}
    return metrics


def computeAllManualBinaryMetrics(predictions: list[int], labels: list[int], label: Optional[str] = None,
                                  prefix: Optional[str] = None) -> dict:
    predictions = np.array(predictions).reshape(-1)
    labels = np.array(labels).reshape(-1)
    if DROP_NAN_BEFORE_COMPUTE_METRICS:
        predictions, labels = dropNanValuesSimultaneously(predictions, labels)
    warnings.filterwarnings("error")
    try:
        false_pos, true_pos, roc_thresholds = sk_metrics.roc_curve(labels, predictions)
        false_pos = false_pos.tolist()
        true_pos = true_pos.tolist()
        roc_thresholds = roc_thresholds.tolist()
    except (ValueError, UndefinedMetricWarning):
        false_pos = true_pos = roc_thresholds = None
    try:
        precisions, recalls, pr_thresholds = sk_metrics.precision_recall_curve(labels, predictions)
        precisions = precisions.tolist()
        recalls = recalls.tolist()
        pr_thresholds = pr_thresholds.tolist()
    except (ValueError, UndefinedMetricWarning):
        precisions = recalls = pr_thresholds = None
    try:
        roc_auc_score = sk_metrics.roc_auc_score(labels, predictions)
    except (ValueError, UndefinedMetricWarning):
        roc_auc_score = -1
    try:
        acc = sk_metrics.accuracy_score(labels, predictions)
    except UndefinedMetricWarning:
        acc = -1
    try:
        prec = sk_metrics.precision_score(labels, predictions)
    except UndefinedMetricWarning:
        prec = -1
    try:
        rec = sk_metrics.recall_score(labels, predictions)
    except UndefinedMetricWarning:
        rec = -1
    try:
        f1 = sk_metrics.f1_score(labels, predictions)
    except UndefinedMetricWarning:
        f1 = -1
    cm = sk_metrics.confusion_matrix(labels, predictions).tolist()
    label = f'_{label}' if label is not None else ''
    prefix = f'_{prefix}' if prefix is not None else ''
    metrics = {prefix + 'acc' + label: acc,
               prefix + 'pre' + label: prec,
               prefix + 'rec' + label: rec,
               prefix + 'f1' + label: f1,
               prefix + 'roc_auc' + label: roc_auc_score,
               prefix + 'graphs' + label: {
                   'roc': {
                       'false_pos': false_pos,
                       'true_pos': true_pos,
                       'thresholds': roc_thresholds,
                   },
                   'pr': {
                       'precisions': precisions,
                       'recalls': recalls,
                       'thresholds': pr_thresholds,
                   },
                   'cm': cm
               }}
    warnings.resetwarnings()
    return metrics
