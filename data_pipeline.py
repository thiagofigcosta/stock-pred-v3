from typing import Union

from crawler import downloadTicker
from enricher import enrich
from hyperparameters import Hyperparameters
from preprocessor import ProcessedDataset, preprocess
from transformer import transform


def runPipeline(ticker: str, start_date: str, end_date: str, configs: Hyperparameters, load_test: bool = True,
                encode: bool = True, get_path: bool = False) -> Union[ProcessedDataset, tuple[ProcessedDataset, str]]:
    """
    A facade to run the entire data pipeline: download, enrich, transform and preprocess.
    :param ticker: the stock or ticker letter code, e.g. GOOG
    :param start_date: data start date, dd/mm/yyy
    :param end_date: data end date, dd/mm/yyy
    :param configs: experiment hyperparameters, necessary to configure the enrichment, transform and preprocess steps
    :param load_test: if true, loads the test portion of the dataset
    :param encode: if true, prepare the data for lstm, creating backwards and forward rolling window for neurons input
    :param get_path: if true, returns the dataset csv path along with the ProcessedDataset
    :return: the processed data in the ProcessedDataset object and maybe the path to the csv file
    """
    dataset_filepath = downloadTicker(ticker, start_date, end_date, configs=configs)
    dataset_filepath = enrich(dataset_filepath, configs=configs)
    dataset_filepath = transform(dataset_filepath, configs=configs)
    processed_data = preprocess(dataset_filepath, encode_rolling_window=encode, load_test=load_test, configs=configs)
    if get_path:
        return processed_data, dataset_filepath
    else:
        return processed_data
