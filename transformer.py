from typing import Union, Optional

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import MinMaxScaler

from hyperparameters import Hyperparameters
from logger import info, verbose
from plotter import plot
from prophet_filepaths import getTickerScalerFilepath, getTransformedTickerFilepath, getTickerFromFilename
from utils_date import dateStrToDateObj, SAVE_DATE_FORMAT
from utils_fs import getBasename, pathExists, removeFileExtension
from utils_persistance import saveObj, loadDataframe, saveDataframe, loadDataframeColumns

DATE_COLUMN = 'Date'
PREDICT_COLUMN = 'close_to_predict'


def getOptimalComponentsToReduce(df: pd.DataFrame, knee_sensibility: int, use_kernel_pca: bool,
                                 kernel_pca_type: str = 'linear', filename: Optional[str] = None) -> KneeLocator:
    verbose(f'Finding the optimal amount of features...')
    if use_kernel_pca:
        kpca_transform = KernelPCA(kernel=kernel_pca_type).fit_transform(df)
        explained_variance = np.var(kpca_transform, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        explained_variance_ratio_sum = np.cumsum(explained_variance_ratio)
    else:
        pca = PCA().fit(df)
        explained_variance_ratio_sum = np.cumsum(pca.explained_variance_ratio_)
    optimum_n_components = KneeLocator(list(range(len(explained_variance_ratio_sum))), explained_variance_ratio_sum,
                                       curve='concave',
                                       direction='increasing', S=knee_sensibility)

    verbose(f'Found the optimal amount of features, reducing from {df.shape[1]} to '
            f'{optimum_n_components.knee}, keeping {round(optimum_n_components.knee_y * 100, 3)}% of '
            f'the variance!')
    plot_data = [
        ('line', [explained_variance_ratio_sum], {'label': '_nolegend_', 'zorder': 1}),
        ('scatter', [[optimum_n_components.knee], [optimum_n_components.knee_y]],
         {'color': 'red', 'label': f'optimum # components ({optimum_n_components.knee})',
          'zorder': 2}),
    ]
    file_label = f'knee_method_{f"kpca-{kernel_pca_type}" if use_kernel_pca else "pca"}'
    plot(plot_data, title='Cumulative explained variance per # of components', x_label='Number of components',
         y_label='Cumulative explained variance', file_label=file_label, subdir='dataset',
         file_prefix=removeFileExtension(getBasename(filename)), file_postfix=False, add_rid_subdir=False)

    return optimum_n_components


def normalizeDataset(df_or_series: Union[pd.DataFrame, pd.Series], feat_range: tuple[float, float],
                     log: bool = True) -> tuple[Union[pd.DataFrame, np.ndarray], MinMaxScaler]:
    if log:
        verbose(f'Normalizing features...')
    scaler = MinMaxScaler(feature_range=feat_range)
    if type(df_or_series) is pd.DataFrame:
        normalized_df_or_series = pd.DataFrame(scaler.fit_transform(df_or_series), columns=df_or_series.columns)
    else:
        normalized_df_or_series = scaler.fit_transform(
            df_or_series.to_numpy().reshape(tuple(list(df_or_series.shape) + [-1])))
    if log:
        verbose(f'Normalized features!')
    return normalized_df_or_series, scaler


def reduceDimensionality(df: pd.DataFrame, optimum_n_components: KneeLocator, use_kernel_pca: bool,
                         kernel_pca_type: str = 'linear',
                         extract_manually_when_pca: bool = True, force_column_rename: bool = True) -> pd.DataFrame:
    verbose(f'Reducing dimensionality with {"Kernel " if use_kernel_pca else ""}PCA...')
    if use_kernel_pca:
        # I believe we can't fetch column names due to expansion caused by the kernel
        kpca = KernelPCA(kernel=kernel_pca_type, n_components=optimum_n_components.knee)
        reduced = kpca.fit_transform(df)
        selected_columns = [f'kPCA Feature {i}' for i in range(optimum_n_components.knee)]
    else:
        pca = PCA(n_components=optimum_n_components.knee)
        reduced = pca.fit_transform(df)
        if force_column_rename:
            selected_columns = [f'PCA Feature {i}' for i in range(optimum_n_components.knee)]
        else:
            # TODO: this is not working
            n_feats = pca.components_.shape[0]  # important since we could have set a percentage
            selected_columns_scores = [np.abs(pca.components_[i]).argmax() for i in range(n_feats)]
            # I cannot guarantee the columns order are right
            selected_columns = [df.columns.tolist()[selected_columns_scores[i]] for i in range(n_feats)]
            if extract_manually_when_pca:
                reduced = df[selected_columns]
    reduced_df = pd.DataFrame(reduced, index=df.index, columns=selected_columns)
    reduced_df['close'] = df['Close']
    verbose('Close price column added!')
    verbose(f'Reduced dimensionality with {"Kernel " if use_kernel_pca else ""}PCA!')
    return reduced_df


def plotDataframe(df: pd.DataFrame, x_column: str, y_columns: Union[str, list[str], set[str], pd.Index],
                  title: str, label: str, y_label: Optional[str] = None, plot_prefix: Optional[str] = None,
                  x_dateformat: Optional[str] = None, legend: bool = True, legend_outside: bool = False):
    plot_data = []
    if x_dateformat is not None:
        x_column_vals = df[x_column].apply(dateStrToDateObj, input_format=x_dateformat)
    else:
        x_column_vals = df[x_column]
    if type(y_columns) is str:
        y_columns = [y_columns]
        if y_label is None:
            y_label = y_columns[0]
    elif type(y_columns) is pd.Index:
        y_columns = list(y_columns)
    if y_label is None:
        y_label = f'{len(y_columns)} Features'
    if x_column in y_columns:
        y_columns.remove(x_column)
    for y_column in y_columns:
        plot_data.append(('line', [x_column_vals, df[y_column]], {'label': f'{y_column}'}))
    plot(plot_data, title=title, x_label=x_column, y_label=y_label, legend=legend,
         legend_outside=legend_outside, resize=legend_outside, subdir='dataset', add_rid_subdir=False,
         file_prefix=plot_prefix, file_postfix=False, file_label=label)


def transform(filepath: str, force: bool = False, configs: Optional[Hyperparameters] = None,
              do_plots: bool = True) -> str:
    filename = getBasename(filepath)
    dst_filepath = getTransformedTickerFilepath(filename)

    if pathExists(dst_filepath) and not force:
        return dst_filepath
    info(f'Transforming dataset `{filename}`...')
    if configs is None:
        configs = Hyperparameters.getDefault()
    df = loadDataframe(filepath)
    feature_columns = df.columns.difference([DATE_COLUMN, PREDICT_COLUMN])
    df_only_features = df[feature_columns]
    ticker = getTickerFromFilename(filename)
    plot_prefix = removeFileExtension(filename)

    if do_plots:
        plotDataframe(df, DATE_COLUMN, PREDICT_COLUMN, f'{ticker} Close price', 'close_price',
                      y_label='Close price (USD)',
                      plot_prefix=plot_prefix, x_dateformat=SAVE_DATE_FORMAT)
        plotDataframe(df, DATE_COLUMN, feature_columns,
                      f'{ticker} All Features before {"Kernel " if configs.pca.use_kernel_pca else ""}PCA',
                      'enriched_feats', plot_prefix=plot_prefix, x_dateformat=SAVE_DATE_FORMAT, legend=False)

    df_only_features, first_scaler = normalizeDataset(df_only_features, configs.pca.pca_norm_range, log=False)

    opt_n_components = getOptimalComponentsToReduce(df_only_features, configs.pca.knee_sensibility,
                                                    configs.pca.use_kernel_pca, configs.pca.kernel_pca_type, filename)

    df_reduced = reduceDimensionality(df_only_features, opt_n_components, configs.pca.use_kernel_pca,
                                      configs.pca.kernel_pca_type)

    verbose(f'Features kept: {df_reduced.columns.tolist()}')

    second_scaler = None
    if configs.dataset.normalize:
        df_reduced, second_scaler = normalizeDataset(df_reduced, configs.dataset.norm_range)

    df_reduced.insert(0, DATE_COLUMN, df[DATE_COLUMN])

    if do_plots:
        title = f'{ticker} All {"normalized " if configs.dataset.normalize else ""}features after ' \
                f'{"Kernel " if configs.pca.use_kernel_pca else ""}PCA'
        plotDataframe(df_reduced, DATE_COLUMN, df_reduced.columns, title, 'reduced_feats', plot_prefix=plot_prefix,
                      x_dateformat=SAVE_DATE_FORMAT, legend_outside=True)

    if configs.dataset.normalize_prediction_feat:
        predict_values, third_scaler = normalizeDataset(df[PREDICT_COLUMN], configs.dataset.norm_range, log=False)
        df_reduced[PREDICT_COLUMN] = predict_values
        if do_plots:
            plotDataframe(df_reduced, DATE_COLUMN, PREDICT_COLUMN, f'{ticker} normalized close price',
                          'norm_close_price', y_label='Close price (USD)', plot_prefix=plot_prefix,
                          x_dateformat=SAVE_DATE_FORMAT)
    else:
        third_scaler = None
        df_reduced[PREDICT_COLUMN] = df[PREDICT_COLUMN]

    info(f'Transformed dataset `{filename}` successfully!')
    saveDataframe(df_reduced, dst_filepath)
    info(f'Saved transformed dataset at {dst_filepath}.')

    scaler_dst_filepath = getTickerScalerFilepath(filename)
    scalers = (first_scaler, second_scaler, third_scaler)
    saveObj(scalers, scaler_dst_filepath)
    info(f'Saved dataset scalers at {scaler_dst_filepath}.')
    return dst_filepath


def loadAmountOfFeaturesFromFile(filepath: str) -> Optional[int]:
    features = loadDataframeColumns(getTransformedTickerFilepath(filepath))
    if DATE_COLUMN in features:
        features.remove(DATE_COLUMN)
    if PREDICT_COLUMN in features:
        features.remove(PREDICT_COLUMN)
    amount = len(features)
    return amount if amount > 0 else None
