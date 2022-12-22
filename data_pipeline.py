from typing import Union

from crawler import downloadTicker
from enricher import enrich
from hyperparameters import Hyperparameters
from preprocessor import ProcessedDataset, preprocess
from transformer import transform


def runPipeline(ticker: str, start_date: str, end_date: str, configs: Hyperparameters, load_test: bool = True,
                encode: bool = True, get_path: bool = False) -> Union[ProcessedDataset, tuple[ProcessedDataset, str]]:
    dataset_filepath = downloadTicker(ticker, start_date, end_date, configs=configs)
    dataset_filepath = enrich(dataset_filepath, configs=configs)
    dataset_filepath = transform(dataset_filepath, configs=configs)
    processed_data = preprocess(dataset_filepath, encode_rolling_window=encode, load_test=load_test, configs=configs)
    if get_path:
        return processed_data, dataset_filepath
    else:
        return processed_data
